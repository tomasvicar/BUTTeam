function [train_names,valid_names] = read_data_names()



config.data_path = 'Z:\CARDIO1\CinC_Challenge_2020\Training_WFDB';
config.label_file_path = '..\Partitioning\data\partition';
config.label_file_name = 'labels.json';
config.partition_file_name = 'partition_82.json';


%% Read labels
labels = json_read(config.label_file_path, config.label_file_name);
partition = json_read(config.label_file_path, config.partition_file_name);

train_names = partition(:).train; 
valid_names = partition(:).validation; 



end




%% Read json file
function json_content = json_read(path, file_name)
    % Read json file
    tmp=[path filesep file_name];
    fid = fopen(tmp, 'r');
    if fid == -1
        error('Cannot read JSON file')
    else
        json_content = fread(fid, inf);
        json_content = char(json_content');
    end
    fclose(fid);
    json_content = jsondecode(json_content);
end
