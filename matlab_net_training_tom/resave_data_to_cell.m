clc;clear all;close all force;
addpath('utils')


data_folder='Z:\CARDIO1\CinC_Challenge_2020\Training_WFDB';

[train_names,valid_names] = read_data_names();



train_data = read_data_single(train_names,data_folder);
train_lbls = read_lbls(train_names,data_folder);

valid_data = read_data_single(valid_names,data_folder);
valid_lbls = read_lbls(valid_names,data_folder);






save('data_tmp.mat','train_data','train_lbls','valid_data','valid_lbls','train_names','valid_names')



tmp=cat(2,train_data{:});
means=mean(tmp,2);
stds=std(tmp,[],2);
clear tmp

train_data=normalize(train_data,means,stds);
valid_data=normalize(valid_data,means,stds);
% train_data = cellfun(@(x) x',train_data,'UniformOutput',0);
% valid_data = cellfun(@(x) x',valid_data,'UniformOutput',0);

% train_lbls = cellfun(@(x) x',train_lbls,'UniformOutput',0);
% valid_lbls = cellfun(@(x) x',valid_lbls,'UniformOutput',0);

train_data = train_data';
valid_data = valid_data';



pato_names={'Normal','AF','I-AVB','LBBB','RBBB','PAC','PVC','STD','STE'};

train_lbls = more_hot_encode(train_lbls,pato_names)';
valid_lbls = more_hot_encode(valid_lbls,pato_names)';







save('data_norm_tmp.mat','train_data','train_lbls','valid_data','valid_lbls','train_names','valid_names','means','stds')




% u1=unique(train_lbls);
% u2=unique(train_lbls);
% je=zeros(1,length(u1));
% for k=1:length(u1)
%     for kk=1:length(u2)
%         if strcmp(u1{k},u2{kk})
%             je(k)=1;
%         end
%     end
% end
% if sum(je)<length(u1)
%     error('ve valid setu je lbl co není v trainsetu')
% end
% 
% 
% class_counts=zeros(1,length(u1));
% for k=1:length(train_lbls)
%     for kk=1:length(u1)
%         if strcmp(train_lbls{k},u1{kk})
%             class_counts(kk)=class_counts(kk)+1;
%         end
%         
%     end
% end
% 
% train_lbls_c=categorical(train_lbls);
% valid_lbls_c=categorical(valid_lbls);



