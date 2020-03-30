classdef globalMax < nnet.layer.Layer
    % Example custom weighted addition layer.


    methods
        function layer = globalMax(name) 
            % layer = weightedAdditionLayer(numInputs,name) creates a
            % weighted addition layer and specifies the number of inputs
            % and the layer name.


            layer.Name = name;


        end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X1, ..., Xn) forwards the input data X1,
            % ..., Xn through the layer and outputs the result Z.
            shape=size(X);
            if length(shape)==2

                Z=max(X,[],2);
                
            else
                
                Z=max(X,[],3);
            end
            Z=reshape(Z,[size(Z) 1]);
            
            
            
        end
    end
end