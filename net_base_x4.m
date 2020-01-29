function net = net_base_x4()

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
    function ResBlock(n)
        net.addLayer(sprintf('relu%d', n-1), reluBlock, {sprintf('sum%d', n-1)}, {sprintf('conv%da', n-1)}, {});   
        net.addLayer(sprintf('conv%d', n), convBlock_c, {sprintf('conv%da', n-1)}, {sprintf('conv%d', n)}, {sprintf('conv_%df',n), sprintf('conv_%db', n)});
        net.addLayer(sprintf('relu%d', n), reluBlock, {sprintf('conv%d', n)}, {sprintf('conv%da', n)}, {});
        net.addLayer(sprintf('conv%d', n+1), convBlock_c, {sprintf('conv%da', n)}, {sprintf('conv%d', n+1)}, {sprintf('conv_%df', n+1), sprintf('conv_%db', n+1)});
        net.addLayer(sprintf('sum%d', n+1), dagnn.Sum(), {sprintf('sum%d', n-1), sprintf('conv%d', n+1)}, {sprintf('sum%d', n+1)});
    end

    function ResSkipBlock(n)
        net.addLayer(sprintf('sc%d', n-1), dagnn.Concat('dim', 3), {sprintf('sum%d', n-1), sprintf('sum%dd', n-1)}, {sprintf('sc%d', n-1)});
        net.addLayer(sprintf('relu%dd', n-1), reluBlock, {sprintf('sc%d', n-1)}, {sprintf('conv%dad', n-1)}, {});   
        net.addLayer(sprintf('dr%d', n-1), convBlock_dr, {sprintf('conv%dad', n-1)}, {sprintf('dr%d', n-1)}, {sprintf('dr_%df', n-1), sprintf('dr_%db', n-1)});
        net.addLayer(sprintf('conv%dd', n), convBlock_c, {sprintf('dr%d', n-1)}, {sprintf('conv%dd', n)}, {sprintf('convd_%df', n), sprintf('convd_%db', n)});
        net.addLayer(sprintf('relu%dd', n), reluBlock, {sprintf('conv%dd', n)}, {sprintf('conv%dad', n)}, {});
        net.addLayer(sprintf('conv%dd', n+1), convBlock_c, {sprintf('conv%dad', n)}, {sprintf('conv%dd', n+1)}, {sprintf('convd_%df', n+1), sprintf('convd_%db', n+1)});
        net.addLayer(sprintf('sum%dd', n+1), dagnn.Sum(), {sprintf('sum%dd', n-1), sprintf('conv%dd', n+1)}, {sprintf('sum%dd', n+1)});
    end

%%%============== First Step ================%%%
net.addLayer('GuidedFilter', dagnn.GuidedFiltering(), {'input'}, {'input_base'})
net.addLayer('Div_Elemwise', dagnn.Div_Elemwise, {'input', 'input_base'}, {'input_detail'})
net.addLayer('cat_b', dagnn.Concat('dim', 3), {'input', 'input_base'}, {'input_b'});
net.addLayer('cat_d', dagnn.Concat('dim', 3), {'input', 'input_detail'}, {'input_d'});

%%%============== Base Layer Pass ================%%%
net.addLayer('conv1', convBlock_input, {'input_b'}, {'sum1'}, {'conv_1f', 'conv_1b'});

for b = 2:2:12 % 6 blocks
    ResBlock(b)
end

%%%============== Detail Layer Pass ================%%%
net.addLayer('conv1d', convBlock_input, {'input_d'}, {'conv1d'}, {'convd_1f', 'convd_1b'});
net.addLayer('relu1d', reluBlock, {'conv1d'}, {'conv1ad'}, {});
net.addLayer('conv2d', convBlock_c, {'conv1ad'}, {'conv2d'}, {'convd_2f', 'convd_2b'});
net.addLayer('relu2d', reluBlock, {'conv2d'}, {'conv2ad'}, {});
net.addLayer('conv3d', convBlock_c, {'conv2ad'}, {'conv3d'}, {'convd_3f', 'convd_3b'});
net.addLayer('sum3d', dagnn.Sum(), {'conv1d', 'conv3d'}, {'sum3d'});

for d = 4:2:12 % 5 blocks
    ResSkipBlock(d)
end

%%%============== Fusion ================%%%
net.addLayer('cat', dagnn.Concat('dim', 3), {sprintf('sum%d', b+1), sprintf('sum%dd', d+1)}, {'cat'});
net.addLayer(sprintf('relu%d',b+1), reluBlock, {'cat'}, {sprintf('conv%da', b+1)}, {});
net.addLayer(sprintf('dr%d',b+1), convBlock_dr, {sprintf('conv%da', b+1)}, {sprintf('dr%d', b+1)}, {sprintf('dr_%df', b+1), sprintf('dr_%db', b+1)});
net.addLayer(sprintf('conv%d',b+2), convBlock_c, {sprintf('dr%d', b+1)}, {sprintf('sum%d', b+2)}, {sprintf('conv_%df', b+2), sprintf('conv_%db', b+2)});

for i = b+3:2:b+21 % 10 blocks
    ResBlock(i)
end

net.addLayer(sprintf('relu%d', i+1), reluBlock, {sprintf('sum%d', i+1)}, {sprintf('conv%da', i+1)}, {});
net.addLayer(sprintf('conv%d', i+2), convBlock_c, {sprintf('conv%da', i+1)}, {sprintf('conv%d', i+2)}, {sprintf('conv_%df' ,i+2), sprintf('conv_%db', i+2)});
net.addLayer(sprintf('relu%d', i+2), reluBlock, {sprintf('conv%d', i+2)}, {sprintf('conv%da', i+2)}, {});
net.addLayer(sprintf('conv%d', i+3), convBlock_scalec, {sprintf('conv%da', i+2)}, {sprintf('conv%d', i+3)}, {sprintf('conv_%df', i+3), sprintf('conv_%db', i+3)});
net.addLayer(sprintf('relu%d', i+3), reluBlock, {sprintf('conv%d', i+3)}, {sprintf('conv%da', i+3)}, {});
net.addLayer('SubPixel_x2', dagnn.SubPixel_Conv(), {sprintf('conv%da', i+3)}, {sprintf('conv%db', i+3)});

net.addLayer(sprintf('conv%d', i+4), convBlock_scalec, {sprintf('conv%db', i+3)}, {sprintf('conv%d', i+4)}, {sprintf('conv_%df', i+4), sprintf('conv_%db', i+4)});
net.addLayer(sprintf('relu%d', i+4), reluBlock, {sprintf('conv%d', i+4)}, {sprintf('conv%da', i+4)}, {});
net.addLayer('SubPixel_x4', dagnn.SubPixel_Conv(), {sprintf('conv%da', i+4)}, {sprintf('conv%db', i+4)});

net.addLayer(sprintf('conv%d', i+5), convBlock_c_end, {sprintf('conv%db', i+4)}, {sprintf('conv%d', i+5)}, {sprintf('conv_%df', i+5), sprintf('conv_%db', i+5)});
net.addLayer('Bicubic', dagnn.Bicubic('scale', 4), {'input'}, {'bic'});
net.addLayer('sum', dagnn.Sum(), {'bic', sprintf('conv%d', i+5)}, {'pred'});

%%%============== Loss ================%%%
net.addLayer('loss', dagnn.PSNRLoss(), {'pred', 'label'}, 'objective');
net.initParams();
end