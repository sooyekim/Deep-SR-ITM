function net = net_full()

%%%============== Initialize ================%%%
net = dagnn.DagNN();
reluBlock = dagnn.ReLU;

scale = 2; % scale factor for SR
ch = 64;
convBlock_input = dagnn.Conv('size', [3 3 6 ch], 'hasBias', true, 'stride', [1,1], 'pad', [1,1,1,1]);
convBlock_dr = dagnn.Conv('size', [1 1 2*ch ch], 'hasBias', true, 'stride', [1,1], 'pad', [0,0,0,0]);
convBlock_scalec = dagnn.Conv('size', [3 3 ch scale*scale*ch], 'hasBias', true, 'stride', [1,1], 'pad', [1,1,1,1]);
convBlock_c = dagnn.Conv('size', [3 3 ch ch], 'hasBias', true, 'stride', [1,1], 'pad', [1,1,1,1]);
convBlock_c_end = dagnn.Conv('size', [3 3 ch 3], 'hasBias', true, 'stride', [1,1], 'pad', [1,1,1,1]);
%%%============== Define ResBlocks ================%%%
    function BaseBlock(n)
        net.addLayer(sprintf('relu%d', n-1), reluBlock, {sprintf('sum%d', n-1)}, {sprintf('conv%da', n-1)}, {});   
        net.addLayer(sprintf('conv%d', n), convBlock_c, {sprintf('conv%da', n-1)}, {sprintf('conv%d', n)}, {sprintf('conv_%df', n), sprintf('conv_%db', n)});
        net.addLayer(sprintf('relu%d', n), reluBlock, {sprintf('conv%d', n)}, {sprintf('conv%da', n)}, {});
        net.addLayer(sprintf('conv%d', n+1), convBlock_c, {sprintf('conv%da', n)}, {sprintf('conv%d', n+1)}, {sprintf('conv_%df', n+1), sprintf('conv_%db', n+1)});
    end

    function SkipBlock(n)
        net.addLayer(sprintf('sc%d', n-1), dagnn.Concat('dim', 3), {sprintf('sum%d', n-1), sprintf('sum%dd', n-1)}, {sprintf('sc%d', n-1)});
        net.addLayer(sprintf('relu%dd', n-1), reluBlock, {sprintf('sc%d', n-1)}, {sprintf('conv%dad', n-1)}, {});   
        net.addLayer(sprintf('dr%d', n-1), convBlock_dr, {sprintf('conv%dad', n-1)}, {sprintf('dr%d', n-1)}, {sprintf('dr_%df', n-1), sprintf('dr_%db', n-1)});
        net.addLayer(sprintf('conv%dd', n), convBlock_c, {sprintf('dr%d', n-1)}, {sprintf('conv%dd', n)}, {sprintf('convd_%df', n), sprintf('convd_%db', n)});
        net.addLayer(sprintf('relu%dd', n), reluBlock, {sprintf('conv%dd', n)}, {sprintf('conv%dad', n)}, {});
        net.addLayer(sprintf('conv%dd', n+1), convBlock_c, {sprintf('conv%dad', n)}, {sprintf('conv%dd', n+1)}, {sprintf('convd_%df', n+1), sprintf('convd_%db', n+1)});
    end

    function ResBlock(n)
        BaseBlock(n)
        net.addLayer(sprintf('sum%d', n+1), dagnn.Sum(), {sprintf('sum%d' ,n-1), sprintf('conv%d', n+1)}, {sprintf('sum%d', n+1)});
    end

    function ResModBlock(n)
        BaseBlock(n)
        net.addLayer(sprintf('convbm%d', n/2-1), convBlock_c, {'convbs3a'}, {sprintf('convbm%d', n/2-1)}, {sprintf('convbm_%df', n/2-1), sprintf('convbm_%db', n/2-1)});
        net.addLayer(sprintf('relubm%d',n/2-1), reluBlock, {sprintf('convbm%d', n/2-1)}, {sprintf('convbm%da', n/2-1)}, {});
        net.addLayer(sprintf('convbm%d', n/2), convBlock_c, {sprintf('convbm%da', n/2-1)}, {sprintf('convbm%d', n/2)}, {sprintf('convbm_%df', n/2), sprintf('convbm_%db', n/2)});
        net.addLayer(sprintf('mulbm%d', n/2), dagnn.Mul_Elemwise(), {sprintf('convbm%d', n/2), sprintf('conv%d', n+1)}, {sprintf('mulbm%d', n/2)});
        net.addLayer(sprintf('sum%d', n+1), dagnn.Sum(), {sprintf('sum%d', n-1), sprintf('mulbm%d', n/2)}, {sprintf('sum%d', n+1)});
    end

    function ResSkipBlock(n)
        SkipBlock(n)
        net.addLayer(sprintf('sum%dd', n+1), dagnn.Sum(), {sprintf('sum%dd', n-1), sprintf('conv%dd', n+1)}, {sprintf('sum%dd', n+1)});
    end

    function ResSkipModBlock(n)
        SkipBlock(n)
        net.addLayer(sprintf('convdm%d', n/2-1), convBlock_c, {'convds3a'}, {sprintf('convdm%d', n/2-1)}, {sprintf('convdm_%df', n/2-1), sprintf('convdm_%db', n/2-1)});
        net.addLayer(sprintf('reludm%d', n/2-1), reluBlock, {sprintf('convdm%d', n/2-1)}, {sprintf('convdm%da', n/2-1)}, {});
        net.addLayer(sprintf('convdm%d', n/2), convBlock_c, {sprintf('convdm%da', n/2-1)}, {sprintf('convdm%d', n/2)}, {sprintf('convdm_%df', n/2), sprintf('convdm_%db', n/2)});
        net.addLayer(sprintf('muldm%d', n/2), dagnn.Mul_Elemwise(), {sprintf('convdm%d', n/2), sprintf('conv%dd', n+1)}, {sprintf('muldm%d', n/2)});
        net.addLayer(sprintf('sum%dd', n+1), dagnn.Sum(), {sprintf('sum%dd', n-1), sprintf('muldm%d', n/2)}, {sprintf('sum%dd', n+1)});
    end
        
%%%============== First Step ================%%%
net.addLayer('GuidedFilter', dagnn.GuidedFiltering(), {'input'}, {'input_base'})
net.addLayer('Div_Elemwise', dagnn.Div_Elemwise, {'input', 'input_base'}, {'input_detail'})
net.addLayer('cat_b', dagnn.Concat('dim',3), {'input', 'input_base'}, {'input_b'});
net.addLayer('cat_d', dagnn.Concat('dim',3), {'input', 'input_detail'}, {'input_d'});

%%%============== Base Layer Pass ================%%%
% shared layers for modulation (SMFb)
net.addLayer('convbs1', convBlock_input, {'input_b'}, {'convbs1'}, {'convbs_1f', 'convbs_1b'});
net.addLayer('relubs1', reluBlock, {'convbs1'}, {'convbs1a'}, {});

net.addLayer('convbs2', convBlock_c, {'convbs1a'}, {'convbs2'}, {'convbs_2f', 'convbs_2b'});
net.addLayer('relubs2', reluBlock, {'convbs2'}, {'convbs2a'}, {});

net.addLayer('convbs3', convBlock_c, {'convbs2a'}, {'convbs3'}, {'convbs_3f', 'convbs_3b'});
net.addLayer('relubs3', reluBlock, {'convbs3'}, {'convbs3a'}, {});

% main
net.addLayer('conv1', convBlock_input, {'input_b'}, {'sum1'}, {'conv_1f', 'conv_1b'});
for b = 2:2:12 % 6 blocks
    if mod(b, 4) == 0
        ResModBlock(b)
    else
        ResBlock(b)
    end
end

%%%============== Detail Layer Pass ================%%%
% shared layers for modulation (SMFd)
net.addLayer('convds1', convBlock_input, {'input_d'}, {'convds1'}, {'convds_1f', 'convds_1b'});
net.addLayer('reluds1', reluBlock, {'convds1'}, {'convds1a'}, {});
net.addLayer('convds2', convBlock_c, {'convds1a'}, {'convds2'}, {'convds_2f', 'convds_2b'});
net.addLayer('reluds2', reluBlock, {'convds2'}, {'convds2a'}, {});
net.addLayer('convds3', convBlock_c, {'convds2a'}, {'convds3'}, {'convds_3f', 'convds_3b'});
net.addLayer('reluds3', reluBlock, {'convds3'}, {'convds3a'}, {});

% main
net.addLayer('conv1d', convBlock_input, {'input_d'}, {'conv1d'}, {'convd_1f', 'convd_1b'});
net.addLayer('relu1d', reluBlock, {'conv1d'}, {'conv1ad'}, {});
net.addLayer('conv2d', convBlock_c, {'conv1ad'}, {'conv2d'}, {'convd_2f', 'convd_2b'});
net.addLayer('relu2d', reluBlock, {'conv2d'}, {'conv2ad'}, {});
net.addLayer('conv3d', convBlock_c, {'conv2ad'}, {'conv3d'}, {'convd_3f', 'convd_3b'});
net.addLayer('sum3d', dagnn.Sum(), {'conv1d', 'conv3d'}, {'sum3d'});

for d = 4:2:12 % 5 blocks
    % modulation
    if mod(d, 4) == 0
        ResSkipModBlock(d)
    else
        ResSkipBlock(d)
    end 
end

%%%============== Fusion ================%%%
net.addLayer('cat', dagnn.Concat('dim', 3), {sprintf('sum%d', b+1), sprintf('sum%dd', d+1)}, {'cat'});
net.addLayer(sprintf('relu%d', b+1), reluBlock, {'cat'}, {sprintf('conv%da', b+1)},{});
net.addLayer(sprintf('dr%d', b+1), convBlock_dr, {sprintf('conv%da', b+1)}, {sprintf('dr%d', b+1)}, {sprintf('dr_%df', b+1), sprintf('dr_%db', b+1)});
net.addLayer(sprintf('conv%d', b+2), convBlock_c, {sprintf('dr%d', b+1)}, {sprintf('sum%d', b+2)}, {sprintf('conv_%df', b+2), sprintf('conv_%db', b+2)});

for i = b+3:2:b+21 % 10 blocks
    ResBlock(i)
end

net.addLayer(sprintf('relu%d', i+1), reluBlock, {sprintf('sum%d', i+1)}, {sprintf('conv%da', i+1)}, {});
net.addLayer(sprintf('conv%d', i+2), convBlock_c, {sprintf('conv%da', i+1)}, {sprintf('conv%d', i+2)}, {sprintf('conv_%df', i+2), sprintf('conv_%db', i+2)});
net.addLayer(sprintf('relu%d', i+2), reluBlock, {sprintf('conv%d', i+2)}, {sprintf('conv%da', i+2)}, {});
net.addLayer(sprintf('conv%d', i+3), convBlock_scalec, {sprintf('conv%da', i+2)}, {sprintf('conv%d', i+3)}, {sprintf('conv_%df', i+3), sprintf('conv_%db', i+3)});
net.addLayer(sprintf('relu%d', i+3), reluBlock, {sprintf('conv%d', i+3)}, {sprintf('conv%da', i+3)}, {});

net.addLayer('SubPixel', dagnn.SubPixel_Conv(), {sprintf('conv%da', i+3)}, {sprintf('conv%db', i+3)});
net.addLayer(sprintf('conv%d', i+4), convBlock_c_end, {sprintf('conv%db', i+3)}, {sprintf('conv%d', i+4)}, {sprintf('conv_%df', i+4), sprintf('conv_%db', i+4)});
net.addLayer('Bicubic', dagnn.Bicubic(), {'input'}, {'bic'});
net.addLayer('sum', dagnn.Sum(), {'bic', sprintf('conv%d', i+4)}, {'pred'});

%%%============== Loss ================%%%
net.addLayer('loss', dagnn.PSNRLoss(), {'pred', 'label'}, 'objective');
net.initParams();
end