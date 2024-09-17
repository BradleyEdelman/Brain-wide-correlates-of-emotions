function data_files = fus_organize_files(param)
% Mace and Gogolla Labs - Max Planck Institute of Neurobiology
% Author: Bradley Edeman
% Date: 10.08.22


%% find all fus files
fus_fold = param.directories.fus;
fus_files = dir([fus_fold '\**\*preprocessed.mat']);
fus_folds = {fus_files.folder}';
fus_files = {fus_files.name}';

% find all behavioral label files
behav_fold = param.directories.behav;
% % % % % % % behav_files = dir([behav_fold '\**\*labels.csv']);
% % % % % % % behav_files = {behav_files.name}';

% define pupil path
pupil_fold = param.directories.pupil;

% check that a behavior label file exists for each fus file
data_files = cell(size(fus_files,1)+1,6);
data_files(1,:) = {'fus_file', 'behav_file', 'behav_file_correct', 'pupil_file', 'vba_file', 'whisking_file'};
data_files_removed = cell(0);
for i_fus = size(fus_files,1):-1:1
    
    % extract mouse number and date from fus file (stored in fold info)
    fus_parts = strsplit(fus_folds{i_fus}, '\');
    
    % build behavior label file names
    behav_file = [fus_parts{6} '_' fus_parts{7} '_labels.csv']; % "create" corresponding behavior label file
    behav_file_corrected = [fus_parts{6} '_' fus_parts{7} '_labels_corrected.mat']; % brad-corrected labels
    
    % extract whisking behavior file names
    whisking_path = [pupil_fold fus_parts{6} '\' fus_parts{7} '\'];
    whisking_files = dir([whisking_path 'whisking_behav_labels.mat']);
    if ~isempty(whisking_files) == 1
    whisking_file = whisking_files.name;
    end
    
    % extract pupil file names
    pupil_path = [pupil_fold fus_parts{6} '\' fus_parts{7} '\'];
    pupil_files = dir([pupil_path '*proc.mat']);
    pupil_file = pupil_files.name;
    
    % extract VBA file names
    vba_files = dir([pupil_path '*results.mat']);
    vba_file = vba_files.name;
    
    % store paired fus and behavior full file paths if both exist
    if exist([behav_fold '\' behav_file],'file')
        data_files{i_fus+1,1} = [fus_folds{i_fus} '\' fus_files{i_fus}];
        data_files{i_fus+1,2} = [behav_fold behav_file];
        data_files{i_fus+1,4} = [pupil_path pupil_file];
        data_files{i_fus+1,5} = [pupil_path vba_file];
    else
        data_files(i_fus+1,:) = [];
        data_files_removed{end+1} = [behav_fold '\' behav_file];
    end
    
    % check separately for corrected behavior files as those must be processed (original behav file must also exist, obviously)
    if exist([behav_fold '\' behav_file],'file') && exist([behav_fold '\' behav_file_corrected],'file')
        data_files{i_fus+1,3} = [behav_fold behav_file_corrected];
    end
    
    if exist([behav_fold '\' behav_file],'file') && exist([whisking_path '\' whisking_file],'file')
        data_files{i_fus+1,6} = [whisking_path whisking_file];
    end
    
end

fprintf('\n %.0f sessions loaded\n', size(data_files,1)-1);
fprintf('\n %.0f sessions removed\n', size(data_files_removed,2));
