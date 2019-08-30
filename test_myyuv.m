clear all;
%%%====== Settings ======%%%
yuv_format = '420'; % YUV file format
SDR_file = './data/test/testset_SDR.yuv'; % input .yuv file
wid = 1920; % width
hei = 1080; % height
num_fr = 28; % number of frames in the YUV file
scale = 2; % scale factor for SR
pred_file = sprintf('./pred/pred_x%d.yuv', scale); % result .yuv file
gpuDevice(1); % GPU ID
%%%======================%%%
addpath('utils');
disp(['Testing for scale ', num2str(scale), '...'])
% initialize
fclose(fopen(pred_file, 'w'));
[fwidth,fheight] = yuv_factor(yuv_format);

% load net
disp('Loading net...')
netstruct = load(sprintf('./net/x%d.mat', scale));
net = dagnn.DagNN.loadobj(netstruct.net);
move(net,'gpu');
net.mode = 'test' ;
pred_index = net.getVarIndex('pred'); 
net.conserveMemory = true;

% test
disp('Testing starts...')
for fr = 1:num_fr
    % read frames
    SDR_YUV = uint8(load_yuv(SDR_file, fr, hei, wid, fheight, fwidth, 'SDR'));
    % normalize
    SDR_YUV = single(SDR_YUV)/255;
    % change type
    SDR_YUV = gpuArray(SDR_YUV);
    % prediction
    net.eval({'input', SDR_YUV});
    pred = gather(net.vars(pred_index).value);
    pred = min(max(pred, 0), 1);
    % save yuv file
    save_yuv(pred*1023, pred_file, hei, wid, fheight, fwidth, 'HDR');
    disp('Frame saved!')
end
disp('Done!')
