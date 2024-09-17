function fus_extract_behavior_bout_videos(data_files, param, results_file)
% Mace Lab - Max Planck Institute of Neurobiology
% Author: Bradley Edeman
% Date: 04.11.21

%% Initialize directories
close all; fclose all;

% extract relevant parameters
behav_videos = param.behavior.videos.names;
rewrite = param.behavior.videos.rewrite;

% define save file for storing behavior bout videos
video_dir = [param.save_fold param.behavior.videos.video_fold];

if exist(results_file,'file') && rewrite == 1
    
    % "rewriting" indicates deleting all videos and files for all behaviors
    if exist(video_dir,'dir'); rmdir(video_dir,'s'); end
    mkdir(video_dir)
    
    load(results_file);

    % load behavior label meanings
    label_meanings = table2array(readtable('J:/Patrick McCarthy/behaviour_predictions/label_meanings.csv','ReadVariableNames',0));
    
    % determine which behavior indices to analyze based on which behaviors were specified in input
    behav_idx = find(contains(label_meanings, behav_videos));
    
    % "open" bout info for easier looping
    B_bout_total_info_expanded = cat(2, B_bout_total_info{:});
    
    for i_behavior = 1:size(behav_idx,1) % for each behavior of interest
        
        % extract bout information for a particular behavior
        behav_bout_info = B_bout_total_info_expanded(behav_idx(i_behavior),:);
        
        % create save directory for the bouts of this behavior
        behav_save_dir = [video_dir label_meanings{behav_idx(i_behavior)} '\'];
        if ~exist(behav_save_dir); mkdir(behav_save_dir); end
        
        % initialize a cell array containing information about each bout for later review and documentation
        bout_param = {'bout #', 'Animal #', 'Session date', 'Bout onset (sec)', 'Bout offset (sec)', 'Manual approval (y/n)'};
        
        bout_count = 1; % keep track of bout count 
        for i_session = 1:size(behav_bout_info,2) % for each session (not all sessions have a bout for the behavior of interest)

            for i_bout = 1:size(behav_bout_info{i_session},2) % for each bout within each session

                % extract current bout indices
                bout_idx_fus = behav_bout_info{i_session}{i_bout}; % bout indices in fus temporal sampling (need to convert back to video temporal sampling)
                bout_idx_fus = [bout_idx_fus(1) bout_idx_fus(end)]; % only keep onset and offset range
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % adjust onset and offset according to pre-set parameters
                bout_idx_fus(1) = bout_idx_fus(1) + param.behavior.videos.start_adjust;
                bout_idx_fus(2) = bout_idx_fus(2) + param.behavior.videos.end_adjust;
                
                % convert fus frames to sec
                bout_idx_sec = round(bout_idx_fus*0.6);
                
                % make sure the video is at least one sec long
                if diff(bout_idx_sec) == 0
                    bout_idx_sec(2) = bout_idx_sec(1) + 1;
                end
                
                % convert sec to video frames
                bout_idx_video = bout_idx_sec*30; % hard code 30 Hz video frame rate

                % find video file that corresponds to current session
                parts = strsplit(data_files{i_session+1,1},'\');
                video_fold = [parts{1} '\' parts{2} '\' parts{3} '\' parts{4} '\' parts{6} '\' parts{7} '\'];
                video_file = dir([video_fold '\**\temp*.avi']);
                video_file = [video_fold video_file.name];

                % load video file
                video_data = VideoReader(video_file);
                frame = read(video_data, bout_idx_video);

                % create gif of current bout
                video_file = [behav_save_dir 'bout_#' num2str(bout_count) '.avi'];
                aviObj = VideoWriter(video_file);
                set(aviObj, 'FrameRate', 30)
                open(aviObj);

                f = figure(34); clf
                for i_frame = 1:size(frame,4)

                    h = imagesc(frame(:,:,:,i_frame));
                    title([label_meanings{behav_idx(i_behavior)} ' bout # ' num2str(bout_count)])
                    axis off; drawnow

                    curr_frame = getframe(f); 
                    writeVideo(aviObj, curr_frame)

                end
                close(aviObj);
%                 fileattrib(video_file,'-w')

                % save information about bouts
                bout_param{1+bout_count,1} = num2str(bout_count);      % bout number
                bout_param{1+bout_count,2} = parts{6};                 % animal ID
                bout_param{1+bout_count,3} = parts{7};                 % session date
                bout_param{1+bout_count,4} = num2str(bout_idx_sec(1)); % bount onset (sec)
                bout_param{1+bout_count,5} = num2str(bout_idx_sec(2)); % bout offset (sec)
                bout_param{1+bout_count,6} = '';                       % manual approval (to be filled out later)

                bout_count = bout_count + 1; % increase bout count
            end
        end

        % save bout information to excel file for each behavior
        review_file = [behav_save_dir '\Bout_param_review.xlsx'];
        if exist(review_file,'file'); delete(review_file); end
        xlswrite(review_file, bout_param)
    end

end    
            