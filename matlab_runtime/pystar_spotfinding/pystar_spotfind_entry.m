function metadata_json = pystar_spotfind_entry(volume_path, config_json)
%PYSTAR_SPOTFIND_ENTRY Python-facing MATLAB spot-finding entrypoint.

step_timer = tic;
cfg = jsondecode(config_json);

vol_zyx = pystar_load_volume_tiff(volume_path);
spots = pystar_spotfind_max3d(vol_zyx, cfg.threshold_rel);

output_path = fullfile(fileparts(volume_path), 'spots_output.csv');
spot_table = array2table(spots, 'VariableNames', {'z', 'y', 'x', 'intensity'});
writetable(spot_table, output_path, 'Delimiter', ',', 'QuoteStrings', false);

metadata = struct();
metadata.round_id = cfg.round_id;
metadata.reference_round = cfg.reference_round;
metadata.channel_id = cfg.channel_id;
metadata.algorithm = cfg.algorithm;
metadata.matlab_method = cfg.matlab_method;
metadata.volume_shape_zyx = cfg.volume_shape_zyx;
metadata.n_spots = size(spots, 1);
metadata.output_path = output_path;
metadata.steps = {struct( ...
    'name', 'pystar_spotfind_max3d', ...
    'duration_ms', toc(step_timer) * 1000.0, ...
    'details', struct('threshold_rel', cfg.threshold_rel))};

metadata_json = jsonencode(metadata);
end
