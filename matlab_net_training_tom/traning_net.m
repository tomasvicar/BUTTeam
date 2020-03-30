clc;clear all;close all force;
addpath('utils')

load('data_norm_tmp.mat')


train_lbls=cat(1,train_lbls{:});
valid_lbls=cat(1,valid_lbls{:});


lbls_count=sum(train_lbls,1);
classWeights=(sum(lbls_count)./lbls_count);
classWeights=classWeights/min(classWeights);
classWeights=ones(size(classWeights));


numResponses = size(train_lbls,2);
featureDimension = size(train_data{1},1);

% numHiddenUnits=10;
% layers = [...
%     sequenceInputLayer(featureDimension,'Name','input','Normalization','none')
%     bilstmLayer(8,'OutputMode','sequence','Name',['lstm' num2str(1)])
%     subsampleLayer(['subsample' num2str(1)])
%     bilstmLayer(8,'OutputMode','sequence','Name',['lstm' num2str(2)])
%     subsampleLayer(['subsample' num2str(2)])
%     bilstmLayer(16,'OutputMode','sequence','Name',['lstm' num2str(3)])
%     subsampleLayer(['subsample' num2str(3)])
%     bilstmLayer(16,'OutputMode','sequence','Name',['lstm' num2str(4)])
%     subsampleLayer(['subsample' num2str(4)])
%     bilstmLayer(32,'OutputMode','sequence','Name',['lstm' num2str(5)])
%     subsampleLayer(['subsample' num2str(5)])
%     bilstmLayer(32,'OutputMode','sequence','Name',['lstm' num2str(6)])
%     globalMax('globalmax')
%     fullyConnectedLayer(numResponses,'Name','fcfinal_final')
%     sigLayer()
%     weightedCEregresionLayer(classWeights)];
% 
% layers = [...
%     sequenceInputLayer(featureDimension,'Name','input','Normalization','none')
%     bilstmLayer(8,'OutputMode','sequence','Name',['lstm' num2str(1)])
%     bilstmLayer(8,'OutputMode','sequence','Name',['lstm' num2str(2)])
%     bilstmLayer(8,'OutputMode','sequence','Name',['lstm' num2str(3)])
%     globalMax('globalmax')
%     fullyConnectedLayer(numResponses,'Name','fcfinal_final')
%     sigLayer()
%     weightedCEregresionLayer(classWeights)];
% 
% layers=layerGraph(layers);


layers = [...
    sequenceInputLayer(featureDimension,'Name','input','Normalization','none')
    bilstmLayer(8,'OutputMode','sequence','Name',['lstm' num2str(1)])
    bilstmLayer(8,'OutputMode','sequence','Name',['lstm' num2str(2)])
    bilstmLayer(8,'OutputMode','sequence','Name',['lstm' num2str(3)])
    bilstmLayer(8,'OutputMode','last','Name',['lstm' num2str(3)])
    fullyConnectedLayer(numResponses,'Name','fcfinal_final')
    sigLayer()
    weightedCEregresionLayer(classWeights)];

layers=layerGraph(layers);
   

% numHiddenUnits = 100;
% numFc1=200;
% numFc2=100;
% blocks=2;
% layers = [sequenceInputLayer(featureDimension,'Name','input','Normalization','none')];
% for k=1:blocks
%     layers = [...
%         layers
%         fullyConnectedLayer(numFc1,'Name',['fc' num2str(k) '0'])
%         bilstmLayer(numHiddenUnits,'OutputMode','sequence','Name',['lstm' num2str(k)])
%         fullyConnectedLayer(numFc1,'Name',['fc' num2str(k) '1'])
%         reluLayer('Name',['r' num2str(k) '1'])
%         dropoutLayer(0.5,'Name',['do' num2str(k) '1'])
%         fullyConnectedLayer(numFc2,'Name',['fc' num2str(k) '2'])
%         reluLayer('Name',['r' num2str(k) '2'])
%         dropoutLayer(0.5,'Name',['do' num2str(k) '2'])
%         fullyConnectedLayer(numFc1,'Name',['fc' num2str(k) '3'])
%         concatenationLayer(1,3,'Name',['cat' num2str(k) ''])];
%     
% end
% layers = [...
%     layers
%     bilstmLayer(numHiddenUnits,'OutputMode','last','Name','lstm_final')
%     fullyConnectedLayer(numFc1,'Name','fc1_final')
%     reluLayer('Name','r1_final')
%     dropoutLayer(0.5,'Name','do1_final')
%     fullyConnectedLayer(numFc2,'Name','fc2_final')
%     reluLayer('Name','r2_final')
%     dropoutLayer(0.5,'Name','do2_final')
%     fullyConnectedLayer(numFc1,'Name','fc3_final')
%     
%     fullyConnectedLayer(numFc1,'Name','fc1_final2')
%     reluLayer('Name','r1_final2')
%     dropoutLayer(0.5,'Name','do1_final2')
%     fullyConnectedLayer(numFc2,'Name','fc2_final2')
%     reluLayer('Name','r2_final2')
%     dropoutLayer(0.5,'Name','do2_fina2')
%     fullyConnectedLayer(numFc1,'Name','fc3_final2')
%     
%     fullyConnectedLayer(numFc1,'Name','fc1_final3')
%     reluLayer('Name','r1_final3')
%     dropoutLayer(0.5,'Name','do1_final3')
%     fullyConnectedLayer(numFc2,'Name','fc2_final3')
%     reluLayer('Name','r2_final3')
%     dropoutLayer(0.5,'Name','do2_fina3')
%     fullyConnectedLayer(numFc1,'Name','fc3_final3')
%     
%     fullyConnectedLayer(numResponses,'Name','fcfinal_final')
%     sigLayer()
%     weightedCEregresionLayer(classWeights)];
%     
% %     softmaxLayer('Name','sm')
% %     diceClassificationLayer('out')];
% 
% layers=layerGraph(layers);
% layers=connectLayers(layers,'input','cat1/in2');
% layers=connectLayers(layers,'input','cat1/in3');
% for k=1:blocks-1
%     layers=connectLayers(layers,['cat' num2str(k) ''],['cat' num2str(k+1) '/in2']);
%     layers=connectLayers(layers,'input',['cat' num2str(k+1) '/in3']);
% end


save_name=['cpt'];
mkdir(save_name)



batch=32;
sp=0;
options = trainingOptions('adam', ...
    'GradientThreshold',1,...
    'L2Regularization', 1e-8, ...
    'InitialLearnRate',1e-2,...
    'GradientDecayFactor',0.9,...
    'SquaredGradientDecayFactor',0.999,...
    'Epsilon',1e-8,...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',2, ...
    'LearnRateDropFactor',0.1, ...
    'ValidationData',{valid_data,valid_lbls}, ...
    'ValidationFrequency',200,...
    'MaxEpochs', 5, ...
    'MiniBatchSize', batch, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress',...
    'SequencePaddingValue',sp,...
    'SequencePaddingDirection','left',...
    'CheckpointPath', save_name);




net = trainNetwork(train_data,train_lbls,layers,options);

save(['model.mat'],'net')







