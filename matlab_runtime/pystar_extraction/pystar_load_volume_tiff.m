function vol = pystar_load_volume_tiff(volume_path)
%PYSTAR_LOAD_VOLUME_TIFF Load a staged multi-page TIFF as [Z, Y, X].

info = imfinfo(volume_path);
num_pages = numel(info);
if num_pages == 0
    error('pystar:extraction:emptyTiff', 'No TIFF pages found: %s', volume_path);
end

height = info(1).Height;
width = info(1).Width;
sample_page = imread(volume_path, 1);
vol = zeros(num_pages, height, width, 'like', sample_page);

for page_index = 1:num_pages
    page = imread(volume_path, page_index);
    vol(page_index, :, :) = page;
end
end
