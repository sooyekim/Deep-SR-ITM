function train_full_net_x4(varargin)
% load data & label
data = load('./data/train/SDR_youtube_80_x4.mat') ;
label = load('./data/train/HDR_youtube_80.mat') ;
imdb.images.data = data.SDR_data;
imdb.images.label = label.HDR_data;
imdb.images.set = cat(2, ones(1, size(data.SDR_data, 4)-500), 2*ones(1, 500));

% set CNN model
net = net_full_x4();
netstruct = load('./net/net_base_x4/net-epoch-200.mat');
net_init = dagnn.DagNN.loadobj(netstruct.net);

for i = 1:150
    % learning rate and weight decay for biases
    if mod(i, 2) == 0
        net.params(i).learningRate = 0.1;
        net.params(i).weightDecay = 0;
    end
    % initialize net_full with pre-trained values of net_base
    if i>6 && i< 17
        net.params(i).value = net_init.params(i-6).value;
    elseif i>20 && i<29
        net.params(i).value = net_init.params(i-10).value;
    elseif i>32 && i<41
        net.params(i).value = net_init.params(i-14).value;
    elseif i>50 && i<63
        net.params(i).value = net_init.params(i-24).value;
    elseif i>66 && i<79
        net.params(i).value = net_init.params(i-28).value;
    elseif i>82 && i<95
        net.params(i).value = net_init.params(i-32).value;
    elseif i>98
        net.params(i).value = net_init.params(i-36).value;
    end
end
net.conserveMemory = true;

% options
opts.solver = @adam;
opts.train.batchSize = 16;
opts.train.continue = true; 
opts.train.gpus = 1;
opts.train.prefetch = false ;
opts.train.expDir = './net/net_full_x4' ; 
opts.train.learningRate = [1e-7*ones(1, 250) 1e-8*ones(1, 10) 1e-9*ones(1, 10)];
opts.train.weightDecay = 0.0005;
opts.train.numEpochs = numel(opts.train.learningRate) ;
opts.train.derOutputs = {'objective', 1} ;
[opts, ~] = vl_argparse(opts.train, varargin) ;

% record
if(~isdir(opts.expDir))
    mkdir(opts.expDir);
end

% call training function
[net,info] = cnn_train_dag(net, imdb, @getBatch, opts) ;

function inputs = getBatch(imdb, batch, opts)
image = imdb.images.data(:, :, :, batch) ;
label = imdb.images.label(:, :, :, batch) ;

image = single(image)/255;
label = single(label)/1023;
inputs = {'input', gpuArray(image), 'label', gpuArray(label)};