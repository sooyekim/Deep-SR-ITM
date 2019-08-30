function YUV = load_yuv(file, frame, height, width, h_factor, w_factor, SDR_HDR)
%%% Load YUV frame of SDR or HDR videos %%%
% 'SDR_HDR' flag should be specified with either 'SDR' or 'HDR'

% get size of U and V
fileId = fopen(file, 'r');
width_h = width*w_factor;
heigth_h = height*h_factor;
% compute factor for framesize
factor = 1+(h_factor*w_factor)*2;
% compute framesize
framesize = width*height;

if strcmp(SDR_HDR,'HDR') == 1
    fseek(fileId,(frame-1)*factor*framesize*2, 'bof');
    % create Y-Matrix
    YMatrix = fread(fileId, width * height, 'uint16');
    YMatrix = int16(reshape(YMatrix,width,height)');
    % create U- and V- Matrix
    if h_factor == 0
        UMatrix = 0;
        VMatrix = 0;
    else
        UMatrix = fread(fileId,width_h * heigth_h, 'uint16');
        UMatrix = int16(UMatrix);
        UMatrix = reshape(UMatrix,width_h, heigth_h).';

        VMatrix = fread(fileId,width_h * heigth_h, 'uint16');
        VMatrix = int16(VMatrix);
        VMatrix = reshape(VMatrix,width_h, heigth_h).';
    end
elseif strcmp(SDR_HDR,'SDR') == 1
    fseek(fileId,(frame-1)*factor*framesize, 'bof');
        % create Y-Matrix
    YMatrix = fread(fileId, width * height, 'uchar');
    YMatrix = int16(reshape(YMatrix,width,height)');
    % create U- and V- Matrix
    if h_factor == 0
        UMatrix = 0;
        VMatrix = 0;
    else
        UMatrix = fread(fileId,width_h * heigth_h, 'uchar');
        UMatrix = int16(UMatrix);
        UMatrix = reshape(UMatrix,width_h, heigth_h).';

        VMatrix = fread(fileId,width_h * heigth_h, 'uchar');
        VMatrix = int16(VMatrix);
        VMatrix = reshape(VMatrix,width_h, heigth_h).';
    end
end
% compose the YUV-matrix:
YUV(1:height,1:width,1) = YMatrix;

if h_factor == 0
    YUV(:,:,2) = 127;
    YUV(:,:,3) = 127;
end
% consideration of the subsampling of U and V
if w_factor == 1
    UMatrix1(:,:) = UMatrix(:,:);
    VMatrix1(:,:) = VMatrix(:,:);
    
elseif w_factor == 0.5
    UMatrix1(1:heigth_h,1:width) = int16(0);
    UMatrix1(1:heigth_h,1:2:end) = UMatrix(:,1:1:end);
    UMatrix1(1:heigth_h,2:2:end) = UMatrix(:,1:1:end);
    
    VMatrix1(1:heigth_h,1:width) = int16(0);
    VMatrix1(1:heigth_h,1:2:end) = VMatrix(:,1:1:end);
    VMatrix1(1:heigth_h,2:2:end) = VMatrix(:,1:1:end);
    
elseif w_factor == 0.25
    UMatrix1(1:heigth_h,1:width) = int16(0);
    UMatrix1(1:heigth_h,1:4:end) = UMatrix(:,1:1:end);
    UMatrix1(1:heigth_h,2:4:end) = UMatrix(:,1:1:end);
    UMatrix1(1:heigth_h,3:4:end) = UMatrix(:,1:1:end);
    UMatrix1(1:heigth_h,4:4:end) = UMatrix(:,1:1:end);
    
    VMatrix1(1:heigth_h,1:width) = int16(0);
    VMatrix1(1:heigth_h,1:4:end) = VMatrix(:,1:1:end);
    VMatrix1(1:heigth_h,2:4:end) = VMatrix(:,1:1:end);
    VMatrix1(1:heigth_h,3:4:end) = VMatrix(:,1:1:end);
    VMatrix1(1:heigth_h,4:4:end) = VMatrix(:,1:1:end);
end

if h_factor == 1
    YUV(:,:,2) = UMatrix1(:,:);
    YUV(:,:,3) = VMatrix1(:,:);
    
elseif h_factor == 0.5
    YUV(1:height,1:width,2) = int16(0);
    YUV(1:2:end,:,2) = UMatrix1(:,:);
    YUV(2:2:end,:,2) = UMatrix1(:,:);
    
    YUV(1:height,1:width,3) = int16(0);
    YUV(1:2:end,:,3) = VMatrix1(:,:);
    YUV(2:2:end,:,3) = VMatrix1(:,:);
    
elseif h_factor == 0.25
    YUV(1:height,1:width,2) = int16(0);
    YUV(1:4:end,:,2) = UMatrix1(:,:);
    YUV(2:4:end,:,2) = UMatrix1(:,:);
    YUV(3:4:end,:,2) = UMatrix1(:,:);
    YUV(4:4:end,:,2) = UMatrix1(:,:);
    
    YUV(1:height,1:width) = int16(0);
    YUV(1:4:end,:,3) = VMatrix1(:,:);
    YUV(2:4:end,:,3) = VMatrix1(:,:);
    YUV(3:4:end,:,3) = VMatrix1(:,:);
    YUV(4:4:end,:,3) = VMatrix1(:,:);
end

fclose(fileId);

