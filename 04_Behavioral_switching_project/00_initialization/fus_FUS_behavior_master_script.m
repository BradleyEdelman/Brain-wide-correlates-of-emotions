%% Analysis pipeline for FUS + BEHAVIOR

% Mace and Gogolla Labs - Max Planck Institute of Neurobiology
% Author: Bradley Edelman
% Date: 03.11.21

% Intialization script written for Paulina Wanken


%% Set paths 
rmpath(genpath('R:\Work - Experiments\Codes\Paulina\fus\'))
addpath(genpath('R:\Work - Experiments\Codes\Paulina\fus\Brad_fus_and_behavior_analysis\package_20220923\'))

% addpath(genpath('C:\Users\bedelman\Documents\GitHub\MaceGogolla_lab\paulina\package\'));
% rmpath(genpath('C:\Users\bedelman\Documents\GitHub\MaceGogolla_lab\paulina\package\package\99_old_or_unused\'));


%% Identify FUS and Behavior data files

    param.directories.fus = 'J:\Paulina Gabriele Wanken\Data\3D\preprocessed_fus\';
    param.directories.behav = 'J:\Paulina Gabriele Wanken\Data\3D\behaviour_predictions\';
    param.directories.pupil = 'J:\Paulina Gabriele Wanken\Data\3D\';

data_files = fus_organize_files(param);

%% Extract fUS and behavior readouts around bouts

    param.plot_session = 0; % 1/0 plot/dont plot results after each session
    
    param.behavior.bout_threshold = [2 1 5 2]; % threshold behavior bouts (frames) for each behavior (active, egress, groom, inactive)
    param.behavior.bout_window = 25; % how many frames to save before and after bout onset
    param.behavior.random_bout = 6; % number of random bouts to select for each session
    param.behavior.label_meanings = {'inactive', 'egress', 'groom', 'whisking'};
%     param.behavior.label_meanings = {'active', 'egress', 'groom', 'inactive'};
    param.behavior.labels = 'whisking'; % 'original', 'corrected' or 'whisking'
    
    % this is where all the current results will get saved to
    param.save_fold = 'J:\Paulina Gabriele Wanken\Data\3D\01_behavior_fus_analysis\behavior_fus_region\package_20220810\';  
    
    param.fus.rewrite = 0; % do you want to re-run the extract behavior code and overwrite saved data (y - 1, n - 0)
    param.fus.space = 'voxel'; % either 'region' or 'voxel' for fus data
    
results_file = fus_extract_fUS_behavior(data_files, param);
    
%% Generate and save bout videos for specified behaviors. Also save an Excel info file for manual approval/rejection of each bout
    
    param.behavior.label_meanings = {'active', 'egress', 'groom', 'inactive'};
    param.behavior.videos.names = {'egress','groom'}; % speficy which behaviors you want to create bout videos for
    param.behavior.videos.rewrite = 0; % do you want to delete all previous bout videos and approval files and redo the analysis (y - 1, n - 0)
    param.behavior.videos.start_adjust = -9; % specify how many frames before start of bout to include in video
    param.behavior.videos.end_adjust = 0; % specify how many frames after end of bout to include in video
    
    param.behavior.videos.video_fold = 'behavior_bout_videos_9_frame_adjustment\'; % specificy folder to store videos
    
fus_extract_behavior_bout_videos(data_files, param, results_file);

%% Correct behavioral labels based on bout video inspection, save new file
    
    param.behavior.label_meanings = {'active', 'egress', 'groom', 'inactive'};
    param.behavior.correct_rewrite = 0;
    
fus_correct_behav_labels(data_files, param)

%% Organize files AGAIN to include the corrected behavioral label files (if they didnt exist before

data_files = fus_organize_files(param);

%%  Extract fUS and behavior readouts around bouts AGAIN, using corrected behavior labels

    param.behavior.labels = 'corrected';
    param.behavior.label_meanings = {'active', 'egress', 'groom', 'inactive'};
    param.fus.space = 'region'; % either 'region' or 'voxel' for fus data
    param.fus.rewrite = 0;
    
results_file = fus_extract_fUS_behavior(data_files, param);

%% Analyze group (total session) data using only approved bouts and plot everything...
    
    param.fus_behavior_compile_rewrite = 1; % do you want to re compile data and overwrite files (y - 1, n - 0)
fus_compile_fus_behavior_data(data_files, param)

%% Perform individual session GLM analysis
    
    param.glm_individual_rewrite = 1; % perform glm on individual sessions again?
fus_behavior_analysis_GLM(data_files, param)

%% Group level GLM analysis and plotting
    
    param.glm_group_rewrite = 1; % perform glm on individual sessions again?
fus_behavior_analysis_GLM_grp(data_files, param)

%% Time-resolved decoding behavior vs random
    
    param.decoding.rewrite = 0;
    param.behavior.labels = 'corrected';
    param.behavior.label_meanings = {'active', 'egress', 'groom', 'inactive'};
% % %     param.fus.space = 'region';
    
fus_behavior_bout_time_decoding_vs_random(param);


%% Time-resolved decoding behavior vs random
    
    param.mvpa.rewrite_mvpa = 0;
    param.mvpa.rewrite_idx = 0;
    param.behavior.labels = 'whisking';
%     param.behavior.label_meanings = {'active', 'egress', 'groom', 'inactive'};
    param.behavior.label_meanings = {'inactive', 'egress', 'groom', 'whisking'};

fus_behavior_bout_time_decoding_MVPA_voxel(data_files, param)
