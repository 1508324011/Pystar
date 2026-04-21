function metadata_json = pystar_register_global_entry(reference_volume_path, moving_volume_path, config_json)
%PYSTAR_REGISTER_GLOBAL_ENTRY Repo-local MATLAB global registration entrypoint for PyStar.

    warning('off', 'all');

    if ~(ischar(config_json) || isstring(config_json))
        error('pystar_register_global_entry:InvalidConfig', 'config_json must be a JSON string');
    end

    config = jsondecode(char(config_json));
    required_fields = {'fov_id', 'round_id', 'reference_round', 'downsample_factor', 'global_max_shift', 'input_volume_dtype'};
    for idx = 1:numel(required_fields)
        field_name = required_fields{idx};
        if ~isfield(config, field_name)
            error('pystar_register_global_entry:MissingField', 'Config must include %s', field_name);
        end
    end
    if isfield(config, 'volume_transfer_mode') && ~strcmp(char(config.volume_transfer_mode), 'temporary_tiff')
        error('pystar_register_global_entry:InvalidConfig', ...
            'Only volume_transfer_mode="temporary_tiff" is supported for MATLAB registration runtime');
    end

    entry_timer = tic;

    load_timer = tic;
    ref_yxz = new_LoadMultipageTiff(reference_volume_path, config.input_volume_dtype, config.input_volume_dtype, false);
    mov_yxz = new_LoadMultipageTiff(moving_volume_path, config.input_volume_dtype, config.input_volume_dtype, false);
    ref_zyx = permute(ref_yxz, [3, 1, 2]);
    mov_zyx = permute(mov_yxz, [3, 1, 2]);

    if ~isequal(size(ref_zyx), size(mov_zyx))
        error('pystar_register_global_entry:ShapeMismatch', ...
            'Reference and moving registration volumes must share the same shape');
    end

    steps = struct('name', {}, 'duration_ms', {}, 'details', {});
    steps(1) = local_build_step( ...
        'new_LoadMultipageTiff', ...
        toc(load_timer) * 1000, ...
        struct('volume_shape_zyx', size(ref_zyx), 'dtype', char(config.input_volume_dtype)) ...
    );

    register_timer = tic;
    [shift_zyx, global_corr, peak_value] = pystar_phasecorr_global_shift( ...
        ref_zyx, ...
        mov_zyx, ...
        config.downsample_factor, ...
        config.global_max_shift ...
    );
    steps(2) = local_build_step( ...
        'pystar_phasecorr_global_shift', ...
        toc(register_timer) * 1000, ...
        struct('downsample_factor', config.downsample_factor, 'peak_value', peak_value) ...
    );

    metadata = struct();
    metadata.backend = 'matlab_extracted';
    metadata.fov_id = double(config.fov_id);
    metadata.round_id = double(config.round_id);
    metadata.reference_round = double(config.reference_round);
    metadata.scope_mode = char(config.scope_mode);
    metadata.volume_shape_zyx = size(ref_zyx);
    metadata.shift_order = 'z_y_x';
    metadata.shift_semantics = 'apply_to_moving_volume';
    metadata.global_shift = double(shift_zyx);
    metadata.global_corr = double(global_corr);
    metadata.peak_value = double(peak_value);
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
