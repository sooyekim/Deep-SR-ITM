classdef Mul_Elemwise < dagnn.ElementWise
%%% Element-wise multiplication layer %%%
%
% Performs element-wise multiplication of inputs{1} and inputs{2}, as
% outputs{1} = inputs{1}.*inputs{2}
% *Back-propagation (backward function) implemented*

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = inputs{1}.*inputs{2} ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = derOutputs{1}.*inputs{2};
      derInputs{2} = derOutputs{1}.*inputs{1};
      derParams = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1} = inputSizes{1} ;
    end

    function rfs = getReceptiveFields(obj)
      numInputs = numel(obj.net.layers(obj.layerIndex).inputs) ;
      rfs.size = [1 1] ;
      rfs.stride = [1 1] ;
      rfs.offset = [1 1] ;
      rfs = repmat(rfs, numInputs, 1) ;
    end

    function obj = Mul_Elemwise(varargin)
      obj.load(varargin) ;
    end
  end
end
