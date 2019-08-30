classdef PSNRLoss < dagnn.Loss
%%% L2 loss with PSNR caculation for forward pass %%%
%
% This loss function back propagates the L2 loss, but calculates
% the PSNR in the forward pass for better analysis of the prediction.
% Therefore, this loss acts the same way as an L2 loss when training the network.
% *Back-propagation (backward function) implemented as L2 loss*

    methods
        function outputs = forward(obj, inputs, params)
            imdff = double(inputs{1}) - double(inputs{2});
            rmse = sqrt(mean(mean(mean(imdff.^2,1),2),3));
            psnr = 20*log10(1/rmse);
            outputs{1} = mean(squeeze(psnr));
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + size(inputs{1},4)*double(gather(outputs{1}))) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs,params, derOutputs)
            Y = gather(bsxfun(@minus,inputs{1},inputs{2}));
            Y(Y>1)= 1;  % x-y>1
            Y(Y<-1) = -1; % y-x<1

            derInputs{1} = gpuArray(bsxfun(@times, derOutputs{1},Y));
            derInputs{2} = [] ;
            derParams = {} ;
        end
        
        function obj = PSNRLoss(varargin)
            obj.load(varargin) ;
        end
    end
end
