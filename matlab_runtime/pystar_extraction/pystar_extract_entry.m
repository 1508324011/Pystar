function metadata_json = pystar_extract_entry(volume_path, coords_path, config_json)
%PYSTAR_EXTRACT_ENTRY Python-facing MATLAB extraction entrypoint.

step_timer = tic;
cfg = jsondecode(config_json);

vol_zyx = pystar_load_volume_tiff(volume_path);
coord_table = readtable(coords_path, 'Delimiter', ',', 'ReadVariableNames', true);
coords_zyx = [coord_table.z, coord_table.y, coord_table.x];
spot_index = coord_table.spot_index;

intensities = pystar_boxsum_extract(vol_zyx, coords_zyx, cfg.box_size_zyx);

output_path = fullfile(fileparts(volume_path), 'extraction_output.csv');
output_table = table(spot_index, intensities, 'VariableNames', {'spot_index', 'intensity'});
writetable(output_table, output_path, 'Delimiter', ',', 'QuoteStrings', false);

metadata = struct();
metadata.round_id = cfg.round_id;
metadata.channel_id = cfg.channel_id;
metadata.method = cfg.method;
metadata.transform_application_mode = cfg.transform_application_mode;
metadata.n_spots = cfg.n_spots;
metadata.box_size_zyx = cfg.box_size_zyx;
metadata.volume_shape_zyx = cfg.volume_shape_zyx;
metadata.output_path = output_path;
metadata.steps = {struct( ...
    'name', 'pystar_boxsum_extract', ...
    'duration_ms', toc(step_timer) * 1000.0, ...
    'details', struct('coords_path', coords_path))};

metadata_json = jsonencode(metadata);
end
