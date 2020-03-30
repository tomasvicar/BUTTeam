clc;clear all;close all force;
addpath('utils')

load('data_norm_tmp.mat')
load(['model.mat'])


lens=cellfun(@(x) size(x,2),train_data);

sp=0;
batch=32;

valid_res={};
for k=1:length(valid_lbls)
    k

    
    tmp=valid_data{k};
    pad_size=max(lens(randperm(length(lens),batch)));
    tmp = padarray(tmp,[0,pad_size],sp,'pre');
    vyss=predict(net,tmp,'MiniBatchSize',1);

    valid_res=[valid_res,vyss];
    drawnow;

end


valid_lbls_vec=cat(1,valid_lbls{:});
valid_res_vec=cat(1,valid_res{:});