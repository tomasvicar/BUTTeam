function lbls=read_lbls(names,data_folder)

lbls=repmat({},[1,length(names)]);
for k=1:length(names)
    
    file_name=[data_folder filesep names{k} '.hea'];
    
    fid = fopen(file_name,'r');
    
    patern='#Dx: ';
    line='';
    
    while ~startsWith(line,patern)
        line= fgetl(fid);
    end
    
    diagnosis=replace(line,patern,'');
    
    fclose(fid);
    
    
    lbls=[lbls,diagnosis];
    
end



end

