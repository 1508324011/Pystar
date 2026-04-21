function intensities = pystar_boxsum_extract(vol_zyx, coords_zyx, box_size_zyx)
%PYSTAR_BOXSUM_EXTRACT Box-sum extraction using 0-based ZYX coordinates.

if ndims(vol_zyx) ~= 3
    error('pystar:extraction:invalidVolume', 'Expected 3D volume, got ndims=%d', ndims(vol_zyx));
end

if size(coords_zyx, 2) ~= 3
    error('pystar:extraction:invalidCoords', 'Expected coords as N x 3 [z y x] matrix');
end

if numel(box_size_zyx) ~= 3
    error('pystar:extraction:invalidBox', 'Expected box_size_zyx to contain 3 integers');
end

[depth, height, width] = size(vol_zyx);
box_size_zyx = double(box_size_zyx(:)');
half_box = floor(box_size_zyx ./ 2);
num_spots = size(coords_zyx, 1);

coords_int = round(double(coords_zyx)) + 1;
intensities = zeros(num_spots, 1, 'single');

for spot_index = 1:num_spots
    cz = coords_int(spot_index, 1);
    cy = coords_int(spot_index, 2);
    cx = coords_int(spot_index, 3);

    z1 = max(1, cz - half_box(1));
    z2 = min(depth, cz + half_box(1));
    y1 = max(1, cy - half_box(2));
    y2 = min(height, cy + half_box(2));
    x1 = max(1, cx - half_box(3));
    x2 = min(width, cx + half_box(3));

    intensities(spot_index) = single(sum(vol_zyx(z1:z2, y1:y2, x1:x2), 'all'));
end
end
