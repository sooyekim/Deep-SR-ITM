classdef Div_Elemwise < dagnn.ElementWise
%%% Element-wise division layer %%%
%
% Performs element-wise division of inputs{1} by inputs{2}, as
% outputs{1} = inputs{1}./(inputs{2}+eps)
% *Back-propagation (backward function) implemented*

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = inputs{1}./(inputs{2}+eps) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = derOutputs{1}.*(1./(inputs{2}+eps));
      derInputs{2} = derOutputs{1}.*(-1./(inputs{2}.^2+eps)).*inputs{1};
      derParams = {} ;
    end

    function obj = Div_Elemwise(varargin)
      obj.load(varargin) ;
    end
  end
end
