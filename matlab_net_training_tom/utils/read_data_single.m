function data=read_data_single(names,data_folder)

data=repmat({},[1,length(names)]);
for k=1:length(names)
    
    file_name=[data_folder filesep names{k} '.mat'];
    
%     val=[];
    
    load(file_name)
    
    if isempty(val)
        error(['cant read mat ' file_name])
        
    end
    
%     data=[data,val];
    data{k}=single(val);
    
end



end

