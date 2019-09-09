clear all;
%%%====== Settings ======%%%
model = 'Deep SR-ITM'; % Deep SR-ITM (ICCV'19) or Multi-purpose CNN (ACCV'18)
yuv_format = '420'; % YUV file format
SDR_file = './data/test/testset_SDR.yuv'; % input .yuv file
HDR_file = './data/test/testset_HDR.yuv'; % GT .yuv file
wid = 3840; % width
hei = 2160; % height
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
if strcmp(model, 'Deep SR-ITM')
    netstruct = load(sprintf('./net/x%d.mat', scale));
elseif strcmp(model, 'Multi-purpose CNN')
    netstruct = load(sprintf('./net/Multi-purpose_CNN_x%d.mat', scale));
end
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
    HDR_YUV = uint16(load_yuv(HDR_file, fr, hei, wid, fheight, fwidth, 'HDR'));
    % normalize
    SDR_YUV = single(SDR_YUV)/255;
    HDR_YUV = single(HDR_YUV)/1023;
    % create LR data
    SDR_LR_YUV = imresize(SDR_YUV, 1/scale);
    % change type
    SDR_LR_YUV = gpuArray(SDR_LR_YUV);
    % prediction
    net.eval({'input', SDR_LR_YUV});
    pred = gather(net.vars(pred_index).value);
    pred = min(max(pred, 0), 1);
    
    %%% Evaluation %%% (comment or uncomment appropriate lines)
    % *some metrics may be slow*
    psnr_all(fr) = psnr(HDR_YUV, pred, 1);
%     ssim_all(fr) = ssim(HDR_YUV, pred); % SSIM
%     mpsnr_all(fr) = mPSNR_HDR(HDR_YUV, pred, -3, 3); % mPSNR
%     msssim_all(fr) = msssim(HDR_YUV*1023, pred*1023, 1023);

    disp(['#', num2str(fr), ' PSNR: ', num2str(psnr_all(fr)), ' dB'])
%     disp(['#', num2str(fr), ' PSNR: ', num2str(psnr_all(fr)), ' dB', ' SSIM: ', num2str(ssim_all(fr)),...
%         ' mPSNR: ', num2str(mpsnr_all(fr)), ' dB', ' MS-SSIM: ', num2str(msssim_all(fr))]);

    % save yuv file
    save_yuv(pred*1023, pred_file, hei, wid, fheight, fwidth, 'HDR');
    disp('Frame saved!')
end
disp(['Average PSNR: ', num2str(mean(psnr_all)), ' dB'])
% disp(['Avg PSNR: ', num2str(mean(psnr_all)), ' dB', ' Avg SSIM: ', num2str(mean(ssim_all)),...
%     ' Avg mPSNR: ', num2str(mean(mpsnr_all)), ' dB', ' Avg MS-SSIM: ', num2str(mean(msssim_all))]);
% REFERENCE: Avg PSNR: 35.5414 dB Avg SSIM: 0.98304 Avg mPSNR: 37.9845 dB Avg MS-SSIM: 0.98382

disp('Done!')
