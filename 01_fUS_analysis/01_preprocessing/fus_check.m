function load_file = fus_check(storage, Load, Save)
load_file = {};

% master storage folder
if ~exist(storage,'dir')
    mkdir(storage);
end

% check that files to be loaded exist
load_file = [];
for i = 1:size(Load,2)

    % load folder
    proc_fold = [storage Load{i} '\'];
    if ~exist(proc_fold,'dir')
        mkdir(proc_fold);
    end

    % load file
    files = dir(proc_fold); files = {files.name}';
    files = files(contains(files,['I_' Load{i}]));
    if isempty(files)
        fprintf(['No files of type: I_' Load{i}])
    end
    
    for j = 1:size(files,1)
        load_file = [load_file; {[proc_fold files{j}]}];
    end
        
%     load_file{i} = [proc_fold 'I_' Load{i} '.mat'];
%     if ~exist(load_file{i},'file')
%         error([proc_fold 'file does not exist'])
%     end

end

% check that save directories exist
for i = 1:size(Save,2)

    % save folder
    proc_fold = [storage Save{i} '\'];
    if ~exist(proc_fold,'dir')
        mkdir(proc_fold)
    end

end