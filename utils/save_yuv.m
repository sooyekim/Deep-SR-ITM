function save_yuv(data, new_file, height, width, h_factor, w_factor, SDR_HDR)
%%% Save YUV frame of SDR or HDR videos %%%
% 'SDR_HDR' flag should be specified with either 'SDR' or 'HDR'

% get size of data
datasize = size(data);
datasizelength = length(datasize);

% open file
fid = fopen(new_file,'a');

% subsampling of U and V
if datasizelength == 2 || h_factor == 0
    %4:0:0
    y(1:height, 1:width) = data(:, :, 1);
elseif datasizelength == 3
    y(1:height, 1:width) = double(data(:, :, 1));
    u(1:height, 1:width) = double(data(:, :, 2));
    v(1:height, 1:width) = double(data(:, :, 3));
    if w_factor == 1
        %4:1:1
        u2 = u;
        v2 = v;
    elseif h_factor == 0.5
        %4:2:0
        u2(1:height/2, 1:width/2) = u(1:2:end, 1:2:end)+u(2:2:end, 1:2:end)+u(1:2:end, 2:2:end)+u(2:2:end, 2:2:end);
        u2                         = u2/4;
        v2(1:height/2, 1:width/2) = v(1:2:end, 1:2:end)+v(2:2:end, 1:2:end)+v(1:2:end, 2:2:end)+v(2:2:end, 2:2:end);
        v2                         = v2/4;
    elseif w_factor == 0.25
        %4:1:1
        u2(1:height, 1:width/4) = u(:, 1:4:end)+u(:, 2:4:end)+u(:, 3:4:end)+u(:, 4:4:end);
        u2                       = u2/4;
        v2(1:height, 1:width/4) = v(:, 1:4:end)+v(:, 2:4:end)+v(:, 3:4:end)+v(:, 4:4:end);
        v2                       = v2/4;
    elseif w_factor == 0.5 && h_factor == 1
        %4:2:2
        u2(1:height, 1:width/2) = u(:, 1:2:end)+u(:, 2:2:end);
        u2                       = u2/2;
        v2(1:height, 1:width/2) = v(:, 1:2:end)+v(:, 2:2:end);
        v2                       = v2/2;
    end
end

if strcmp(SDR_HDR,'HDR')
    fwrite(fid,uint16(y'),'uint16'); % write Y-Data

    if h_factor ~= 0
        % write U- and V-Data if not 4:0:0 format
        fwrite(fid, uint16(u2'), 'uint16');
        fwrite(fid, uint16(v2'), 'uint16');
    end
elseif strcmp(SDR_HDR, 'SDR')
        fwrite(fid, uint8(y'), 'uchar'); % write Y-Data

    if h_factor ~= 0
        % write U- and V-Data if not 4:0:0 format
        fwrite(fid, uint8(u2'), 'uchar');
        fwrite(fid, uint8(v2'), 'uchar');
    end
end

fclose(fid);
