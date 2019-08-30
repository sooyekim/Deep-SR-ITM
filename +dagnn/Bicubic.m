classdef Bicubic < dagnn.ElementWise
%%% Bicubic resizing layer %%%
%
% Performs bicubic resizing on inputs{1} and stores it in outputs{1}.
% outputs{1} =imresize(input, obj.scale,'bicubic');
% *Back-propagation (backward function) not implemented*

    properties
        scale=2;
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            outputs{1} =imresize(inputs{1}, obj.scale, 'bicubic');
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = 0*inputs{1} ;
            derParams = {} ;
        end
        
        function obj = Bicubic(varargin)
            obj.load(varargin);
            obj.scale=obj.scale;
        end
    end
end
