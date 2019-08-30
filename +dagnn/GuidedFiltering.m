classdef GuidedFiltering < dagnn.ElementWise
%%% Guided filtering layer %%%
%
% Performs 'self' guided filtering as specified in [1].
% outputs{1} is the self guided filtered result of inputs{1}.
% *Back-propagation (backward function) not implemented*
%
% [1] Kaiming He et al. "Guided Image Filtering", ECCV, 2010.

  properties
    r=5
    eps=0.01
  end

  methods
    function outputs = forward(obj, inputs, params)
        input=inputs{1};
        
        H=(1/(obj.r*obj.r))*ones(obj.r,obj.r);

        meanI=imfilter(input,H,'replicate','same');

        var=imfilter(input.*input,H,'replicate','same')-meanI.*meanI;
        a=var./(var+obj.eps);
        b=meanI-a.*meanI;

        meana=imfilter(a,H,'replicate','same');
        meanb=imfilter(b,H,'replicate','same');

        outputs{1}=meana.*input+meanb;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        % No back propagation needed for Deep SR-ITM
        derInputs{1}=0*derOutputs{1};
        derParams = {};
    end
        

    function obj = GuidedFiltering(varargin)
      obj.load(varargin) ;
      obj.r=obj.r;
      obj.eps=obj.eps;
    end
  end
end
