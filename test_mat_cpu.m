clear all;
%%%====== Settings ======%%%
model = 'Multi-purpose CNN'; % 'Deep SR-ITM' (ICCV'19) or 'Multi-purpose CNN' (ACCV'18)
SDR_file = './data/test/testset_SDR.mat'; % input .mat file
HDR_file = './data/test/testset_HDR.mat'; % GT .mat file
scale = 2; % scale factor for SR
pred_file = sprintf('./pred/pred_x%d.mat', scale); % result .mat file
%%%======================%%%
addpath('utils');
% load data
disp(['Testing for scale ', num2str(scale), '...'])
disp('Loading file...')
SDR = load(SDR_file);
HDR = load(HDR_file);
data = SDR.SDR;
label = HDR.HDR;

% initialize
psnr_all = zeros(1, size(data, 4));
ssim_all = zeros(1, size(data, 4));
mpsnr_all = zeros(1, size(data, 4));
msssim_all = zeros(1, size(data, 4));
pred = single(zeros(size(data)));

% load net
disp('Loading net...')
if strcmp(model, 'Deep SR-ITM')
    netstruct = load(sprintf('./net/x%d.mat', scale));
elseif strcmp(model, 'Multi-purpose CNN')
    netstruct = load(sprintf('./net/Multi-purpose_CNN_x%d.mat', scale));
end
if strcmp(model, 'Deep SR-ITM')
    netstruct.net.layers(172).type='dagnn.SubPixel_Conv_cpu';
elseif strcmp(model, 'Multi-purpose CNN')
    netstruct.net.layers(37).type='dagnn.SubPixel_Conv_cpu';
end
net = dagnn.DagNN.loadobj(netstruct.net);
move(net,'cpu');
net.mode = 'test' ;
pred_index = net.getVarIndex('pred'); 
net.conserveMemory = true;

% test
disp('Testing starts...')
for fr = 1:size(data, 4)
    % read frames
    SDR_YUV = single(data(:, :, :, fr));
    HDR_YUV = single(label(:, :, :, fr));
    % normalize
    SDR_YUV = SDR_YUV/255;
    HDR_YUV = HDR_YUV/1023;
    % create LR data
    SDR_LR_YUV = imresize(SDR_YUV, 1/scale);
    % prediction
    net.eval({'input', SDR_LR_YUV});
    pred_fr = net.vars(pred_index).value;
    pred(:, :, :, fr) = min(max(pred_fr, 0), 1);
    
    %%% Evaluation %%% (comment or uncomment appropriate lines)
    % *some metrics may be slow*
    psnr_all(fr) = psnr(HDR_YUV, pred(:, :, :, fr), 1); % PSNR
%     ssim_all(fr) = ssim(HDR_YUV, pred(:, :, :, fr)); % SSIM
%     mpsnr_all(fr) = mPSNR_HDR(HDR_YUV, pred(:, :, :, fr), -3, 3); % mPSNR
%     msssim_all(fr) = msssim(HDR_YUV*1023, pred(:, :, :, fr)*1023, 1023);

    disp(['#', num2str(fr), ' PSNR: ', num2str(psnr_all(fr)), ' dB'])
%     disp(['#', num2str(fr), ' PSNR: ', num2str(psnr_all(fr)), ' dB', ' SSIM: ', num2str(ssim_all(fr)),...
%         ' mPSNR: ', num2str(mpsnr_all(fr)), ' dB', ' MS-SSIM: ', num2str(msssim_all(fr))]);
end
disp(['Average PSNR: ', num2str(mean(psnr_all)), ' dB'])
% disp(['Avg PSNR: ', num2str(mean(psnr_all)), ' dB', ' Avg SSIM: ', num2str(mean(ssim_all)),...
%     ' Avg mPSNR: ', num2str(mean(mpsnr_all)), ' dB', ' Avg MS-SSIM: ', num2str(mean(msssim_all))]);
% REFERENCE: Avg PSNR: 35.5966 dB Avg SSIM: 0.9833 Avg mPSNR: 38.0449 dB Avg MS-SSIM: 0.9843
% REFERENCE x4: Avg PSNR: 33.69 dB Avg SSIM: 0.9657 Avg mPSNR: 36.13 Avg MS-SSIM: 0.9754

% save as .mat file
disp('Saving...')
save(pred_file, 'pred', '-v7.3');
disp('Done!')
