classdef weightedCEregresionLayer < nnet.layer.RegressionLayer
               
    properties
        % Vector of weights corresponding to the classes in the training
        % data
        ClassWeights
    end

    methods
        function layer = weightedCEregresionLayer(classWeights)
            % layer = weightedClassificationLayer(classWeights) creates a
            % weighted cross entropy loss layer. classWeights is a row
            % vector of weights corresponding to the classes in the order
            % that they appear in the training data.
            % 
            % layer = weightedClassificationLayer(classWeights, name)
            % additionally specifies the layer name. 

            % Set class weights
            layer.ClassWeights = classWeights;

            % Set layer name

            layer.Name = 'weightedCEregresionLayer';
                


            % Set layer description
            layer.Description = 'Weighted cross entropy';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the weighted cross
            % entropy loss between the predictions Y and the training
            % targets T.

            N = size(Y,4);
            Y = squeeze(Y);
            T = squeeze(T);
            W = layer.ClassWeights;
    
            loss = -sum(W*(T.*log(Y)+(1-T).*log(1-Y)))/N;
            
            
            
        end
    end
end