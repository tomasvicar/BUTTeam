classdef diceClassificationLayer < nnet.layer.ClassificationLayer
    % This layer implements the generalized dice loss function for training
    % semantic segmentation networks.
    
    properties(Constant)
        % Small constant to prevent division by zero. 
        Epsilon = 1e-8;
    end
    
    methods
        
        function layer = diceClassificationLayer(name)
            % layer =  dicePixelClassificationLayer(name) creates a Dice
            % pixel classification layer with the specified name.
            
            % Set layer name.          
            layer.Name = name;
            
            % Set layer description.
            layer.Description = 'Dice loss';
        end
        
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the Dice loss between
            % the predictions Y and the training targets T.   
            
            Y=reshape(Y,size(Y,1),[]);
            T=reshape(T,size(T,1),[]);

            % Weights by inverse of region size.
            W = 1 ./ (sum(T,2).^2+ layer.Epsilon);
            
            intersection = sum(Y.*T,2);
            union = sum(Y.^2 + T.^2,2);          
            
            numer = 2*sum(W.*intersection,1) + layer.Epsilon;
            denom = sum(W.*union,1) + layer.Epsilon;
            
            % Compute Dice score.
            dice = numer./denom;
            
            % Return average Dice loss.
%             N = size(Y,2);
%             loss = sum((1-dice))/N;
            loss = (1-dice);
            
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % dLdY = backwardLoss(layer, Y, T) returns the derivatives of
            % the Dice loss with respect to the predictions Y.
            
            size0=size(Y);
            Y=reshape(Y,size(Y,1),[]);
            T=reshape(T,size(T,1),[]);
            
            
            % Weights by inverse of region size.
            W = 1 ./ (sum(T,2).^2+ layer.Epsilon);
            
            intersection = sum(Y.*T,2);
            union = sum(Y.^2 + T.^2,2);
     
            numer = 2*sum(W.*intersection,1) + layer.Epsilon;
            denom = sum(W.*union,1) + layer.Epsilon;
            
%             N = size(Y,2);
      
%             dLdY = (2*W.*Y.*numer./denom.^2 - 2*W.*T./denom)./N;
            dLdY = (2*W.*Y.*numer./denom.^2 - 2*W.*T./denom);
            
            dLdY=reshape(dLdY,size0(1),size0(2),size0(3));
            
        end
    end
end