function [shift_zyx, global_corr, peak_value] = pystar_phasecorr_global_shift(ref_zyx, mov_zyx, downsample_factor, max_shift)
%PYSTAR_PHASECORR_GLOBAL_SHIFT Compute whole-volume global shift for PyStar.
%   Inputs are expected in [Z, Y, X] order and the returned shift is the
%   shift to apply to the moving volume in [dz, dy, dx] order.

    if nargin < 3 || isempty(downsample_factor)
        downsample_factor = 1;
    end
    if nargin < 4 || isempty(max_shift)
        max_shift = 200;
    end

    if ndims(ref_zyx) ~= 3 || ndims(mov_zyx) ~= 3
        error('pystar_phasecorr_global_shift:InvalidInput', 'Expected two 3D volumes in [Z,Y,X] order');
    end
    if ~isequal(size(ref_zyx), size(mov_zyx))
        error('pystar_phasecorr_global_shift:ShapeMismatch', 'Reference and moving volumes must share the same shape');
    end

    downsample_factor = max(1, round(double(downsample_factor)));
    ref_small = single(ref_zyx);
    mov_small = single(mov_zyx);
    if downsample_factor > 1
        ref_small = ref_small(:, 1:downsample_factor:end, 1:downsample_factor:end);
        mov_small = mov_small(:, 1:downsample_factor:end, 1:downsample_factor:end);
    end

    ref_small = ref_small - mean(ref_small(:), 'omitnan');
    mov_small = mov_small - mean(mov_small(:), 'omitnan');

    cps = fftn(ref_small) .* conj(fftn(mov_small));
    denom = abs(cps);
    denom(denom == 0) = 1;
    corr_volume = abs(ifftn(cps ./ denom));

    [peak_value, linear_idx] = max(corr_volume(:));
    [peak_z, peak_y, peak_x] = ind2sub(size(corr_volume), linear_idx);
    signed_shift = double([peak_z - 1, peak_y - 1, peak_x - 1]);
    dims_small = double(size(corr_volume));
    half_dims = floor(dims_small ./ 2);
    wrap_mask = signed_shift > half_dims;
    signed_shift(wrap_mask) = signed_shift(wrap_mask) - dims_small(wrap_mask);

    shift_zyx = -signed_shift;
    if downsample_factor > 1
        shift_zyx(2:3) = shift_zyx(2:3) * downsample_factor;
    end

    shift_candidate = round(shift_zyx);
    corr_candidate = local_mip_corr(ref_zyx, mov_zyx, shift_candidate);
    corr_flipped = local_mip_corr(ref_zyx, mov_zyx, -shift_candidate);
    if corr_flipped > corr_candidate
        shift_candidate = -shift_candidate;
        corr_candidate = corr_flipped;
    end

    if norm(shift_candidate) > double(max_shift)
        warning('pystar_phasecorr_global_shift:LargeShift', ...
            'Estimated global shift exceeds max_shift=%d (norm=%.3f)', ...
            round(double(max_shift)), norm(shift_candidate));
    end

    shift_zyx = double(shift_candidate);
    global_corr = double(corr_candidate);
    peak_value = double(peak_value);
end


function corr_value = local_mip_corr(ref_zyx, mov_zyx, shift_zyx)
    shifted = local_apply_integer_shift_zyx(mov_zyx, shift_zyx);
    ref_mip = squeeze(max(ref_zyx, [], 1));
    shifted_mip = squeeze(max(shifted, [], 1));
    corr_value = local_safe_corr(double(ref_mip), double(shifted_mip));
end


function shifted = local_apply_integer_shift_zyx(volume, shift_zyx)
    dims = size(volume);
    shift_zyx = round(double(shift_zyx));
    shifted = zeros(dims, 'like', volume);

    [src_z, dst_z] = local_axis_ranges(dims(1), shift_zyx(1));
    [src_y, dst_y] = local_axis_ranges(dims(2), shift_zyx(2));
    [src_x, dst_x] = local_axis_ranges(dims(3), shift_zyx(3));

    if isempty(src_z) || isempty(src_y) || isempty(src_x)
        return;
    end

    shifted(dst_z, dst_y, dst_x) = volume(src_z, src_y, src_x);
end


function [src_idx, dst_idx] = local_axis_ranges(axis_length, shift_amount)
    if shift_amount >= axis_length || shift_amount <= -axis_length
        src_idx = [];
        dst_idx = [];
        return;
    end

    if shift_amount >= 0
        src_start = 1;
        src_end = axis_length - shift_amount;
        dst_start = 1 + shift_amount;
        dst_end = axis_length;
    else
        src_start = 1 - shift_amount;
        src_end = axis_length;
        dst_start = 1;
        dst_end = axis_length + shift_amount;
    end

    if src_start > src_end || dst_start > dst_end
        src_idx = [];
        dst_idx = [];
        return;
    end

    src_idx = src_start:src_end;
    dst_idx = dst_start:dst_end;
end


function corr_value = local_safe_corr(img_a, img_b)
    a = img_a(:);
    b = img_b(:);

    if all(a == a(1)) || all(b == b(1))
        corr_value = 0.0;
        return;
    end

    corr_mat = corrcoef(a, b);
    corr_value = corr_mat(1, 2);
    if isnan(corr_value)
        corr_value = 0.0;
    end
end
