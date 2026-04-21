function metadata_json = pystar_register_local_demons_entry(reference_volume_path, moving_volume_path, config_json)
%PYSTAR_REGISTER_LOCAL_DEMONS_ENTRY Repo-local MATLAB local demons entrypoint for PyStar.

    warning('off', 'all');

    if ~(ischar(config_json) || isstring(config_json))
        error('pystar_register_local_demons_entry:InvalidConfig', 'config_json must be a JSON string');
    end

    config = jsondecode(char(config_json));
    required_fields = {
        'fov_id', ...
        'round_id', ...
        'reference_round', ...
        'input_volume_dtype', ...
        'iterations', ...
        'accumulated_field_smoothing', ...
        'flow_output_path', ...
        'expected_flow_shape_zyx', ...
        'global_shift_already_applied'
    };

    for idx = 1:numel(required_fields)
        field_name = required_fields{idx};
        if ~isfield(config, field_name)
            error('pystar_register_local_demons_entry:MissingField', 'Config must include %s', field_name);
        end
    end

    if isfield(config, 'volume_transfer_mode') && ~strcmp(char(config.volume_transfer_mode), 'temporary_tiff')
        error('pystar_register_local_demons_entry:InvalidConfig', ...
            'Only volume_transfer_mode="temporary_tiff" is supported for MATLAB registration runtime');
    end

    entry_timer = tic;

    load_timer = tic;
    ref_yxz = new_LoadMultipageTiff(reference_volume_path, config.input_volume_dtype, config.input_volume_dtype, false);
    mov_yxz = new_LoadMultipageTiff(moving_volume_path, config.input_volume_dtype, config.input_volume_dtype, false);

    if ~isequal(size(ref_yxz), size(mov_yxz))
        error('pystar_register_local_demons_entry:ShapeMismatch', ...
            'Reference and moving registration volumes must share the same shape');
    end

    expected_flow_shape_zyx = double(config.expected_flow_shape_zyx);
    expected_shape_yxz = [expected_flow_shape_zyx(2), expected_flow_shape_zyx(3), expected_flow_shape_zyx(1)];
    if ~isequal(size(ref_yxz), expected_shape_yxz)
        error('pystar_register_local_demons_entry:UnexpectedVolumeShape', ...
            'Loaded registration volume shape does not match expected_flow_shape_zyx');
    end

    steps = struct('name', {}, 'duration_ms', {}, 'details', {});
    steps(1) = local_build_step( ...
        'new_LoadMultipageTiff', ...
        toc(load_timer) * 1000, ...
        struct('volume_shape_yxz', size(ref_yxz), 'dtype', char(config.input_volume_dtype)) ...
    );

    pyd_level = floor(log2(size(ref_yxz, 3)));
    if pyd_level == 0
        pyd_level = 1;
    end
    if isfield(config, 'pyramid_levels') && ~isempty(config.pyramid_levels)
        configured_levels = double(config.pyramid_levels);
        if configured_levels > 0
            pyd_level = configured_levels;
        end
    end

    register_timer = tic;
    displacement_field_yxz = imregdemons( ...
        mov_yxz, ...
        ref_yxz, ...
        double(config.iterations), ...
        'PyramidLevels', double(pyd_level), ...
        'AccumulatedFieldSmoothing', double(config.accumulated_field_smoothing), ...
        'DisplayWaitbar', false ...
    );
    displacement_field_yxz = single(displacement_field_yxz);
    steps(2) = local_build_step( ...
        'imregdemons', ...
        toc(register_timer) * 1000, ...
        struct( ...
            'iterations', double(config.iterations), ...
            'accumulated_field_smoothing', double(config.accumulated_field_smoothing), ...
            'pyramid_levels', double(pyd_level) ...
        ) ...
    );

    save_timer = tic;
    save(char(config.flow_output_path), 'displacement_field_yxz', '-v7');
    steps(3) = local_build_step( ...
        'save_local_flow_mat', ...
        toc(save_timer) * 1000, ...
        struct('flow_output_path', char(config.flow_output_path)) ...
    );

    metadata = struct();
    metadata.backend = 'matlab_extracted';
    metadata.fov_id = double(config.fov_id);
    metadata.round_id = double(config.round_id);
    metadata.reference_round = double(config.reference_round);
    metadata.scope_mode = char(config.scope_mode);
    metadata.volume_shape_zyx = double(config.expected_flow_shape_zyx);
    metadata.flow_output_path = char(config.flow_output_path);
    metadata.flow_storage_format = 'mat_v7';
    metadata.flow_variable = 'displacement_field_yxz';
    metadata.flow_layout = 'y_x_z_components';
    metadata.flow_component_order = 'dx_dy_dz';
    metadata.flow_semantics = 'apply_to_moving_volume';
    metadata.flow_composition = 'residual_after_global_shift';
    metadata.global_shift_already_applied = logical(config.global_shift_already_applied);
    metadata.flow_shape_yxz_component = double(size(displacement_field_yxz));
    metadata.mean_abs_displacement = double(mean(abs(displacement_field_yxz(:))));
    if isfield(config, 'compute_tile_index')
        metadata.compute_tile_index = double(config.compute_tile_index);
    end
    if isfield(config, 'compute_tile_grid_position_yx')
        metadata.compute_tile_grid_position_yx = double(config.compute_tile_grid_position_yx);
    end
    if isfield(config, 'compute_tile_grid_shape_yx')
        metadata.compute_tile_grid_shape_yx = double(config.compute_tile_grid_shape_yx);
    end
    if isfield(config, 'compute_tile_origin_zyx')
        metadata.compute_tile_origin_zyx = double(config.compute_tile_origin_zyx);
    end
    if isfield(config, 'compute_tile_shape_zyx')
        metadata.compute_tile_shape_zyx = double(config.compute_tile_shape_zyx);
    end
    if isfield(config, 'compute_tile_write_origin_zyx')
        metadata.compute_tile_write_origin_zyx = double(config.compute_tile_write_origin_zyx);
    end
    if isfield(config, 'compute_tile_write_shape_zyx')
        metadata.compute_tile_write_shape_zyx = double(config.compute_tile_write_shape_zyx);
    end
    if isfield(config, 'compute_tile_write_offset_zyx')
        metadata.compute_tile_write_offset_zyx = double(config.compute_tile_write_offset_zyx);
    end
    metadata.steps = steps;
    metadata.total_duration_ms = toc(entry_timer) * 1000;
    metadata.matlab_version = version;

    metadata_json = jsonencode(metadata);
end


function step = local_build_step(name, duration_ms, details)
    step = struct();
    step.name = name;
    step.duration_ms = duration_ms;
    step.details = details;
end
