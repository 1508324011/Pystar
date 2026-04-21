function metadata_json = pystar_preprocess_entry(input_path, sub_dir, output_dir, config_json)
%PYSTAR_PREPROCESS_ENTRY Repo-local MATLAB preprocessing entrypoint for PyStar.

    warning('off', 'all');

    if ~(ischar(config_json) || isstring(config_json))
        error('pystar_preprocess_entry:InvalidConfig', 'config_json must be a JSON string');
    end

    config = jsondecode(char(config_json));
    if ~isfield(config, 'fov_id')
        error('pystar_preprocess_entry:MissingField', 'Config must include fov_id');
    end
    if ~isfield(config, 'round_ids') || isempty(config.round_ids)
        error('pystar_preprocess_entry:MissingField', 'Config must include non-empty round_ids');
    end
    if ~isfield(config, 'seq_channels') || isempty(config.seq_channels)
        error('pystar_preprocess_entry:MissingField', 'Config must include non-empty seq_channels');
    end

    if exist(output_dir, 'dir') ~= 7
        mkdir(output_dir);
    end

    entry_timer = tic;
    load_timer = tic;
    output_imgs = test_LoadImageStacks_zf( ...
        input_path, ...
        sub_dir, ...
        config.round_ids, ...
        config.seq_channels, ...
        config.expected_z_slices, ...
        config.loader_input_format, ...
        config.loader_output_dtype, ...
        false ...
    );

    steps = struct('name', {}, 'duration_ms', {}, 'details', {});
    step_idx = 1;
    steps(step_idx) = local_build_step( ...
        'test_LoadImageStacks_zf', ...
        toc(load_timer) * 1000, ...
        struct('round_count', numel(config.round_ids), 'channel_count', numel(config.seq_channels)) ...
    ); %#ok<AGROW>
    step_idx = step_idx + 1;

    if isfield(config, 'apply_min_max_normalize') && config.apply_min_max_normalize
        step_timer = tic;
        output_imgs = new_MinMaxNorm(output_imgs);
        steps(step_idx) = local_build_step('new_MinMaxNorm', toc(step_timer) * 1000, struct()); %#ok<AGROW>
        step_idx = step_idx + 1;
    end

    if isfield(config, 'equalize_methods') && ~isempty(config.equalize_methods)
        equalize_methods = local_normalize_string_list(config.equalize_methods);
        for method_idx = 1:numel(equalize_methods)
            curr_method = equalize_methods{method_idx};
            step_timer = tic;
            output_imgs = new_EqualizeHist3D(output_imgs, curr_method);
            steps(step_idx) = local_build_step( ...
                'new_EqualizeHist3D', ...
                toc(step_timer) * 1000, ...
                struct('method', curr_method) ...
            ); %#ok<AGROW>
            step_idx = step_idx + 1;
        end
    end

    if isfield(config, 'morphology') && isfield(config.morphology, 'enabled') && config.morphology.enabled
        morph = config.morphology;
        step_timer = tic;
        output_imgs = new_MorphologicalReconstruction( ...
            output_imgs, ...
            char(morph.method), ...
            morph.radius, ...
            morph.height ...
        );
        steps(step_idx) = local_build_step( ...
            'new_MorphologicalReconstruction', ...
            toc(step_timer) * 1000, ...
            struct('method', char(morph.method), 'radius', morph.radius, 'height', morph.height) ...
        ); %#ok<AGROW>
        step_idx = step_idx + 1;
    end

    round_ids = reshape(double(config.round_ids), 1, []);
    seq_channels = reshape(double(config.seq_channels), 1, []);
    output_files = cell(1, numel(round_ids) * numel(seq_channels));
    file_idx = 1;
    save_timer = tic;

    for round_pos = 1:numel(round_ids)
        round_id = round_ids(round_pos);
        for channel_pos = 1:numel(seq_channels)
            channel_id = seq_channels(channel_pos);
            curr_stack = output_imgs(:,:,:,channel_pos,round_pos);
            output_path = fullfile( ...
                output_dir, ...
                sprintf('clean_fov_%d_round_%d_ch_%d.tif', config.fov_id, round_id, channel_id) ...
            );
            SaveSingleTiff(uint8(curr_stack), output_path);
            output_files{file_idx} = output_path;
            file_idx = file_idx + 1;
        end
    end

    steps(step_idx) = local_build_step( ...
        'SaveSingleTiff', ...
        toc(save_timer) * 1000, ...
        struct('output_file_count', numel(output_files)) ...
    ); %#ok<AGROW>

    metadata = struct();
    metadata.backend = 'matlab_extracted';
    metadata.fov_id = double(config.fov_id);
    metadata.sub_dir = sub_dir;
    metadata.round_ids = round_ids;
    metadata.seq_channels = seq_channels;
    metadata.output_shape = size(output_imgs);
    metadata.output_files = output_files;
    metadata.steps = steps;
    metadata.total_duration_ms = toc(entry_timer) * 1000;
    metadata.matlab_version = version;
    if isfield(config, 'skipped_steps')
        metadata.skipped_steps = config.skipped_steps;
    end

    metadata_json = jsonencode(metadata);
end


function step = local_build_step(name, duration_ms, details)
    step = struct();
    step.name = name;
    step.duration_ms = duration_ms;
    step.details = details;
end


function values = local_normalize_string_list(raw_value)
    if isempty(raw_value)
        values = {};
        return;
    end

    if ischar(raw_value) || isstring(raw_value)
        values = cellstr(string(raw_value));
        return;
    end

    if iscell(raw_value)
        values = cellfun(@char, raw_value, 'UniformOutput', false);
        return;
    end

    error('pystar_preprocess_entry:InvalidConfig', 'equalize_methods must be a string, string array, or cell array');
end
