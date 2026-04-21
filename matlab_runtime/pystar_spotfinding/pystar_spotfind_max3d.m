function spots = pystar_spotfind_max3d(vol_zyx, intensity_threshold)
%PYSTAR_SPOTFIND_MAX3D MATLAB max3d-style local-max detector.
% Input volume is [Z, Y, X]. Output rows are [z, y, x, intensity] in 0-based coordinates.

if ndims(vol_zyx) ~= 3
    error('pystar:spotfinding:invalidVolume', 'Expected 3D volume, got ndims=%d', ndims(vol_zyx));
end

% MATLAB image-processing / regionprops3 conventions assume a volume laid out
% as [Y, X, Z]. PyStar stages spot-finding inputs as [Z, Y, X], so permute
% into MATLAB-friendly order before running local-max detection and centroid
% extraction. The returned centroid is then mapped back into PyStar [z, y, x].
vol_yxz = permute(vol_zyx, [2, 3, 1]);

max_mask = imregionalmax(vol_yxz, 26);
threshold_value = intensity_threshold * 255;
candidate_mask = max_mask & (vol_yxz > threshold_value);

cc = bwconncomp(candidate_mask, 26);
if cc.NumObjects == 0
    spots = zeros(0, 4, 'single');
    return;
end

stats = regionprops3(cc, vol_yxz, 'Centroid', 'MaxIntensity');
centroids_xyz = stats.Centroid;
intensities = single(stats.MaxIntensity);

coords_zyx_zero_based = single([
    centroids_xyz(:, 3) - 1, ...
    centroids_xyz(:, 2) - 1, ...
    centroids_xyz(:, 1) - 1 ...
]);

spots = [coords_zyx_zero_based, intensities];
end
