config.data_path = '../data/cinc2020';
config.label_file_path = '';
config.label_file_name = 'labels.json';
config.partition_file_name = 'partition_64.json';


%% Read labels
labels = json_read(config.label_file_path, config.label_file_name);
partition = json_read(config.label_file_path, config.partition_file_name);

labels = containers.Map(fieldnames(labels), {partition.train, partition.validation});
partition = containers.Map(fieldnames(partition), {partition.train, partition.validation});


%% Read json file
% fds = fileDatastore(config.data_path, 'ReadFcn', @mat_file_read, 'FileExtensions','.mat');



%% Read json file
function json_content = json_read(path, file_name)
    % Read json file
    fid = fopen(fullfile(path, file_name), 'r');
    if fid == -1
        error('Cannot read JSON file')
    else
        json_content = fread(fid, inf);
        json_content = char(json_content');
    end
    fclose(fid);
    json_content = jsondecode(json_content);
end


%% Read mat file
function [data, userdata, done] = mat_file_read(filename)
    
    done = false;
    try
        raw_data = load(filename);
    catch
        
    end
    data = raw_data.val;
    done = true;
    
end