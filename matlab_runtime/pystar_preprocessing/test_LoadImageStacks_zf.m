function [output_imgs, dims] = test_LoadImageStacks_zf(inputPath, sub_dir, round_ids, channel_ids, expected_z_slices, input_format, output_format, useGPU)
%TEST_LOADIMAGESTACKS_ZF Reviewable extracted loader for PyStar preprocessing runtime.

    warning('off', 'all');

    if nargin < 6 || isempty(input_format)
        input_format = 'uint8';
    end
    if nargin < 7 || isempty(output_format)
        output_format = 'uint8';
    end
    if nargin < 8
        useGPU = false;
    end

    round_ids = reshape(double(round_ids), 1, []);
    channel_ids = reshape(double(channel_ids), 1, []);
    Nround = numel(round_ids);
    Nchannel = numel(channel_ids);

    if Nround == 0
        error('test_LoadImageStacks_zf:InvalidInput', 'round_ids must not be empty');
    end
    if Nchannel == 0
        error('test_LoadImageStacks_zf:InvalidInput', 'channel_ids must not be empty');
    end

    output_imgs = [];

    for r_idx = 1:Nround
        round_id = round_ids(r_idx);
        fprintf('Loading round %d...\n', round_id);

        for c_idx = 1:Nchannel
            channel_id = channel_ids(c_idx);
            curr_path = local_resolve_round_channel_path(inputPath, sub_dir, round_id, channel_id);
            curr_img = new_LoadMultipageTiff(curr_path, input_format, output_format, useGPU);
            curr_img = local_adjust_z_depth(curr_img, expected_z_slices);

            if isempty(output_imgs)
                output_imgs = zeros( ...
                    size(curr_img, 1), ...
                    size(curr_img, 2), ...
                    expected_z_slices, ...
                    Nchannel, ...
                    Nround, ...
                    output_format ...
                );
            end

            output_imgs(:,:,:,c_idx,r_idx) = curr_img;
        end
    end

    dims = size(output_imgs);
end


function curr_path = local_resolve_round_channel_path(inputPath, sub_dir, round_id, channel_id)
    round_dir = sprintf('round%02d', round_id);
    glob_patterns = {
        sprintf('*_ch%02d.tif', channel_id), ...
        sprintf('*_ch%d.tif', channel_id) ...
    };

    matches = [];
    for pattern_idx = 1:numel(glob_patterns)
        curr_matches = dir(fullfile(inputPath, round_dir, sub_dir, glob_patterns{pattern_idx}));
        if ~isempty(curr_matches)
            matches = curr_matches;
            break;
        end
    end

    if isempty(matches)
        error( ...
            'test_LoadImageStacks_zf:MissingInput', ...
            'Missing TIFF for round %d channel %d under %s/%s', ...
            round_id, channel_id, inputPath, sub_dir ...
        );
    end

    if numel(matches) ~= 1
        error( ...
            'test_LoadImageStacks_zf:AmbiguousInput', ...
            'Expected exactly one TIFF for round %d channel %d under %s/%s, found %d', ...
            round_id, channel_id, inputPath, sub_dir, numel(matches) ...
        );
    end

    curr_path = fullfile(matches(1).folder, matches(1).name);
end


function adjusted = local_adjust_z_depth(curr_img, expected_z_slices)
    if nargin < 2 || isempty(expected_z_slices)
        adjusted = curr_img;
        return;
    end

    expected_z_slices = double(expected_z_slices);
    if size(curr_img, 3) == expected_z_slices
        adjusted = curr_img;
        return;
    end

    if size(curr_img, 3) > expected_z_slices
        error( ...
            'test_LoadImageStacks_zf:UnexpectedZDepth', ...
            'Input TIFF contains %d z-slices, expected at most %d', ...
            size(curr_img, 3), expected_z_slices ...
        );
    end

    adjusted = curr_img;
    adjusted(:,:,end+1:expected_z_slices) = zeros( ...
        size(curr_img, 1), ...
        size(curr_img, 2), ...
        expected_z_slices - size(curr_img, 3), ...
        class(curr_img) ...
    );
end
