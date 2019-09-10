classdef SubPixel_Conv < dagnn.ElementWise
%%% Sub-pixel convolution layer %%%
%
% Performs sub-pixel convolution or pixel shuffle as specified in [1].
% From an input of size [H, W, C, N] stored in inputs{1}, this layer
% produces the output of size [scale*H, scale*W, C/(scale*scale), N] in outputs{1}.
% *Back-propagation (backward function) implemented*
%
% [1] Wenzhe Shi et al. "Real-Time Single Image and Video Super-Resolution
% Using an Efficient Sub-Pixel Convolutional Neural Network", CVPR, 2016.

  properties
    scale=2;
  end

  methods
    function outputs = forward(obj, inputs, params)
        input=inputs{1};
        output=gpuArray(zeros(size(input,1)*obj.scale,size(input,2)*obj.scale,size(input,3)/obj.scale/obj.scale,size(input,4)));
        for channel = 1:size(input,3)
            ch=floor((channel-1)/obj.scale/obj.scale)+1;
            c=mod(channel,obj.scale*obj.scale);
            if c==0, c=obj.scale*obj.scale; end
            q=floor((c-1)/obj.scale)+1;
            r=mod(c, obj.scale);
            if r==0, r=obj.scale; end          
            output(q:obj.scale:end,r:obj.scale:end,ch,:)=input(:,:,channel,:);
        end
        outputs{1}=gpuArray(single(output));
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        output=derOutputs{1};
        input=gpuArray(zeros(size(output,1)/obj.scale,size(output,2)/obj.scale,size(output,3)*obj.scale*obj.scale,size(output,4)));
        for channel = 1:size(input,3)
            ch=floor((channel-1)/obj.scale/obj.scale)+1;
            c=mod(channel,obj.scale*obj.scale);
            if c==0, c=obj.scale*obj.scale; end
            q=floor((c-1)/obj.scale)+1;
            r=mod(c, obj.scale);
            if r==0, r=obj.scale; end            
            input(:,:,channel,:)=output(q:obj.scale:end,r:obj.scale:end,ch,:);
        end
      derInputs{1} = gpuArray(single(input));
      derParams = {} ;
    end

    function obj = SubPixel_Conv(varargin)
      obj.load(varargin) ;
      obj.scale=obj.scale;
    end
  end
end
