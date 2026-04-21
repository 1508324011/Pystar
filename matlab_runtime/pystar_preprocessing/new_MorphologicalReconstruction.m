function input_img = new_MorphologicalReconstruction( input_img, method, varargin )
%MorphologicalReconstruction

    p = inputParser;
    
    defaultRadius = 2;
    defaultHeight = 3;

    addRequired(p,'input_img');
    addRequired(p,'method');
    addOptional(p,'radius',defaultRadius);
    addOptional(p,'height',defaultHeight);

    parse(p, input_img, method, varargin{:});
    
    Nround = size(input_img, 5);
    Nchannel = size(input_img, 4);
    Nslice = size(input_img, 3);
    
    switch method
        case "2d"
            se = strel('disk', p.Results.radius);

            for r=1:Nround
                tic
                fprintf(sprintf("Processing Round %d...", r));

                for c=1:Nchannel
                    curr_channel = input_img(:,:,:,c,r);
                    for z=1:Nslice
                        curr_slice = curr_channel(:,:,z);
                        marker = imerode(curr_slice, se);
                        obr = imreconstruct(marker, curr_slice);
                        curr_out = curr_slice - obr;
                        curr_out = imsubtract(imadd(curr_out, imtophat(curr_out, se)), imbothat(curr_out, se));
                        curr_channel(:,:,z) = curr_out;
                    end
                    input_img(:,:,:,c,r) = uint8(curr_channel);
                end
                fprintf(sprintf('[time = %.2f s]\n', toc));
            end 
   
        case "2d_thres"
            se = strel('disk', p.Results.radius);

            for r=1:Nround
                tic
                fprintf(sprintf("Processing Round %d...", r));

                for c=1:Nchannel
                    curr_channel = input_img(:,:,:,c,r);
                    for z=1:Nslice
                        curr_slice = curr_channel(:,:,z);
                        marker = imerode(curr_slice, se);
                        obr = imreconstruct(marker, curr_slice);
                        curr_out = curr_slice - obr;
                        
                        curr_bw = curr_out > 0 & curr_out < 80;
                        curr_out(curr_bw) = 0;
                        
                        curr_out = im2double(curr_out);
                        curr_max = max(curr_out, [], 'all');
                        curr_min = min(curr_out, [], 'all');
                        curr_out = (curr_out - curr_min) ./ (curr_max - curr_min);
                        curr_out = uint8(curr_out .* 255);
    
                        curr_channel(:,:,z) = curr_out;
                    end
                    input_img(:,:,:,c,r) = uint8(curr_channel);
                end
                fprintf(sprintf('[time = %.2f s]\n', toc));
            end 
            
        case "3d"
            ms = offsetstrel('ball', p.Results.radius, p.Results.height);
            se = strel('sphere', 2);
            
            for r=1:Nround 
                tic
                fprintf(sprintf("Processing Round %d...", r));

                for c=1:Nchannel
                    curr_channel = input_img(:,:,:,c,r);
                    marker = imerode(curr_channel, ms);
                    obr = imreconstruct(marker, curr_channel);
                    curr_out = curr_channel - obr;
                    mask = imbinarize(curr_out,0.06);

                    bw = imopen(mask, se);
                    curr_out(~bw) = 0;

                    curr_out = imsubtract(imadd(curr_out, imtophat(curr_out, ms)), imbothat(curr_out, ms));
                    
                    input_img(:,:,:,c,r) = uint8(curr_out);
                end
                fprintf(sprintf('[time = %.2f s]\n', toc));
            end
            
    end
    
end
