classdef sigLayer < nnet.layer.Layer

    properties
        % (Optional) Layer properties.

        % Layer properties go here.
    end

    properties (Learnable)
        % (Optional) Layer learnable parameters.

        % Layer learnable parameters go here.
    end
    
    methods
        function layer = sigLayer()
            % (Optional) Create a myLayer.
            % This function must have the same name as the layer.

            % Layer constructor function goes here.
            layer.Name = "sigmoid";

            % Set layer description.
            layer.Description = "sigmoid";
            
            
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer    -    Layer to forward propagate through
            %         X        -    Input data
            % Output:
            %         Z        -    Output of layer forward function
            
            % Layer forward function for prediction goes here.
            Z=1./(1+exp(-X));
            
        end

        function [Z, memory] = forward(layer, X)
            % (Optional) Forward input data through the layer at training
            % time and output the result and a memory value.
            %
            % Inputs:
            %         layer  - Layer to forward propagate through
            %         X      - Input data
            % Outputs:
            %         Z      - Output of layer forward function
            %         memory - Memory value for backward propagation

            % Layer forward function for training goes here.
            Z=1./(1+exp(-X));
            memory=[];
        end

        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
            % Backward propagate the derivative of the loss function through 
            % the layer.
            %
            % Inputs:
            %         layer             - Layer to backward propagate through
            %         X                 - Input data
            %         Z                 - Output of layer forward function            
            %         dLdZ              - Gradient propagated from the deeper layer
            %         memory            - Memory value from forward function
            % Outputs:
            %         dLdX              - Derivative of the loss with respect to the
            %                             input data
            %         dLdW1, ..., dLdWn - Derivatives of the loss with respect to each
            %                             learnable parameter
            
            % Layer backward function goes here.
            dLdX=1./(1+exp(-X)).*(1-1./(1+exp(-X))).*dLdZ;
            
        end
    end
end