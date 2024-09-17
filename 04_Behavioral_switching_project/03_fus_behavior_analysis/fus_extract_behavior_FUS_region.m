function [results, results_file] = fus_extract_behavior_FUS_region(data_files, param)
% Mace and Gogolla Labs - Max Planck Institute of Neurobiology
% Author: Bradley Edeman
% Date: 03.11.21

%% 
% extract relevant parameters
bout_threshold = param.bout_threshold;
bout_window = param.bout_window;
save_fold = param.save_fold;
rewrite = param.fus_behavior.rewrite;
space = param.fus_behavior.space;

% define save path/file for storing bout behavior and fus information
save_dir = [save_fold 'behavior_fus_region\'];
if ~exist(save_dir,'dir'); mkdir(save_dir); end
results_file = [save_dir 'behavior_fus_region_' space '_08102022.mat'];

% execute loop if the data file doesnt exist OR if it does exist and the rewrite flag is set to 1
if ~exist(results_file,'file') || exist(results_file,'file') && rewrite == 1
    %% Analyze and store one session at a time

    % load behavior label meanings
    label_meanings = table2array(readtable('J:/Patrick McCarthy/behaviour_predictions/label_meanings.csv','ReadVariableNames',0));
    
    % INITIALIZE LARGE STORAGE VARIABLES
    I_bout_total = cell(1,size(data_files,1)); % time-locked fus data for each behavior for all sessions 
    I_bout_total_during = cell(1,size(data_files,1)); % time-locked behavior data for each beahvior for all sessions
    
    B_bout_total = cell(1,size(data_files,1)); % brain activity during each behavioral bout, independent of duration, for all sessions 
    B_bout_total_info = cell(1,size(data_files,1)); % bout info during each behavior for all sessions
    
    P_bout_total = cell(1,size(data_files,1)); % pupil diameter summed/averaged across each behavioral bout
    P_bout_total_during = cell(1,size(data_files,1)); % pupil diameter during each behavioral bout
    P_whole_recording = cell(size(data_files,1),1); % pupil diameter for entire recording
    
    V_bout_total = cell(1,size(data_files,1)); % VBA data summed/averaged across each behavioral bout
    V_bout_total_during = cell(1,size(data_files,1)); % VBA data during each behavioral bout
    V_whole_recording = cell(size(data_files,1),1); % VBA data for entire recording
   
    mouse_names = cell(size(data_files,1),1); % mouse labels
    session_dates = cell(size(data_files,1),1); % session labels
    L_bout_total = cell(1,size(data_files,1)); % full behavior labels (1-4) during each behavioral bout
    
    
    % For all sessions across all mice
    for i_session = 1:size(data_files,1) 
        i_session
        
        % harcoded extraction of mouse name and session date
        mouse_names{i_session,:} = data_files{i_session+1,1}(1, 53:58);
        session_dates{i_session,:}= data_files{i_session+1,1}(1, 60:67);
   
        % load fus data
        load(data_files{i_session+1,1}); 
        
        % load brain region segmentation
        % % needs to be in loop since atlas is saved with fus
        [new_seg, new_name] = fus_custom_segment_PW([], RefAtlas, 0); % load/create segmentation
        [nx, nz, ny] = size(new_seg);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % SEGMENTATION
        
        % mask new segmentation rather than data
        points_in = double(points_in); % define mask
        points_in(points_in == 0) = nan;
        new_seg = new_seg.*points_in; % apply mask
        
        % linearize new segmentation and fus for easier indexing
        new_seg = reshape(new_seg, nx*nz*ny, []);
        I_signal = reshape(I_signal, nx*ny*nz, []);
        
        if strcmp(space,'region')  % segment time series
            I_final = zeros(size(new_name,1),size(I_signal,2));
            for i_reg = 1:size(new_name,1)
                reg_idx = find(new_seg == i_reg); % all indices of current region
                I_final(i_reg,:) = nanmean(I_signal(reg_idx,:),1); % average time series across all region voxels
            end
        else % leave data in voxel space
            I_final = I_signal;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % BEHAVIOR LABELING
        
        % load behavior labels
        labels = load(data_files{i_session+1,2}); % load behavior labels
        labels = labels(:); %before corretion this and line above was just labels = labels (:);
        
        % ADJUST LABELS ACCORDING TO PATRICK NOTES
        labels = [zeros(100,1); labels]; % add 100 blank frames to the beginning
        labels(end-99:end) = 0; % set last 100 indices to zero (patrick)
        
        % bin behavior labels into fus sampling rate
        fs_fus = 1/metadata.dt; % sampling rate of fus
        fs_behav = 30; % sampling rate of video
        oversampling = ceil(fs_behav/fs_fus); % number of video frames that fit into a single ultrasound frame

        % ensure beahvior label time series has sufficient number of frames for binning
        REM = rem(size(labels,1), oversampling);
        % append or remove extra nan values to ensure same number of bins as fus frames
        if size(labels,1)/oversampling > size(I_final,2) 
            labels = labels(1:end-REM); % if too many frames, remove extras
        elseif size(labels,1)/oversampling < size(I_final,2)
            labels = [labels; nan(oversampling - REM,1)]; % if too few frames, add difference to make a full bin
        end

        % reshape behavior labels into bins
        labels = reshape(labels, oversampling, []);
        % find most common (mode) behavior during each bin
        labels = mode(labels,1);
        % interpolate across any frames that are nan
        labels = fillmissing(labels,'linear',2);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % PUPIL DATA
        
        % load pupil data
        pupil_data = load(data_files{i_session+1,4});
        pupil = pupil_data.pupil{1,1}.area_smooth;
        pupil = movmean(pupil,3);
        
        % bin pupil labels into fus sampling rate
        fs_pupil = 30; % sampling rate of video
        oversampling = ceil(fs_pupil/fs_fus); % number of pupil data points that fit into a single ultrasound frame
        
        % ensure pupil time series has sufficient number of frames for binning
        REM = rem(size(pupil,2), oversampling);
        % append or remove extra nan values to ensure same number of bins as fus frames
        if size(pupil,2)/oversampling > size(I_final,2) 
            pupil = pupil(1:end-REM); % if too many frames, remove extras
        elseif size(pupil,1)/oversampling < size(I_final,2)
            pupil = [pupil; nan(oversampling - REM,1)]; % if too few frames, add difference to make a full bin
        end

        % reshape pupil data into bins
        pupil = reshape(pupil, oversampling, []);
        % here average instead of mode
        pupil = mean(pupil,1);
        % interpolate across any frames that are nan
        pupil = fillmissing(pupil,'linear',2);
        pupil = pupil';
        
        P_whole_recording {i_session} = pupil;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % VBA DATA
        
        vba_data = load(data_files{i_session+1,5});
        vba = vba_data.data.Laser ;
        
        fs_vba = 1000; % sampling rate of vba
        oversampling = ceil(fs_vba/fs_fus); % number of vba data points that fit into a single ultrasound frame
        
        % ensure VBA time series has sufficient number of frames for binning
        REM = rem(size(vba,2), oversampling);
        % append or remove extra nan values to ensure same number of bins as fus frames
        if size(vba,2)/oversampling > size(I_final,2) 
            vba = vba(1:end-REM); % if too many frames, remove extras
        elseif size(vba,1)/oversampling < size(I_final,2)
            vba = [vba; nan(oversampling - REM,1)]; % if too few frames, add difference to make a full bin
        end
        
        % convert VBA output to distance
        vba = (100 - 0) / (5-1) * (vba - 5) + 100; % from 1-5.0 V to 0-100 mm
        vba = (max(vba) - vba + min(vba));
        vba = vba';
        
        % reshape vba time series into bins
        vba = reshape(vba, oversampling, []);
        % here average instead of mode
        vba = mean(vba,1);
        % interpolate across any frames that are nan
        vba = fillmissing(vba,'linear',2);
        vba = vba';
        
        V_whole_recording {i_session} = vba;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % plot all brain region and behavior label data across time
        fus_plot_brain_regions_and_behavior_labels(new_name, I_final, labels, param);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % INDIVIDUAL BEHAVIOR ANALYSIS
        
        % Initialize session-specific storage for each behavior
        I_bout_session = cell(4,1); % time-locked fus data for each behavior for current session 
        I_bout_session_during = cell(4,1); % brain activity during each behavioral bout, independent of duration, for current session 
        
        B_bout_session = cell(4,1); % time-locked behavior data for each behavior for current session 
        B_bout_session_info = cell(4,1); % bout info during each behavior for current session 
        
        P_bout_session = cell(4,1);
        P_bout_session_norm = cell(4,1); % time-locked pupil data for each behavior for current session (normalized)
        P_bout_session_raw = cell(4,1); % time-locked behavior data for each behavior for current session (raw)
        P_bout_session_during = cell(4,1); % Pupil data summed/averaged across each behavioral bout in session
        
        L_bout_session = cell(4,1);
        
        V_bout_session = cell (4,1);
        V_bout_session_norm = cell(4,1); % time-locked vba data for each behavior for current session (normalized)
        V_bout_session_raw = cell(4,1); % time-locked vba data for each behavior for current session (raw)
        V_bout_session_during = cell(4,1); % VBA data summed/averaged across each behavioral bout in session
        
        for i_behavior = 1:size(label_meanings,1)

            % find all onsets of current behavior
            labels_behavior = labels;
            
            % binarize labels for current behavior
            labels_behavior(labels_behavior ~= i_behavior) = 0;
            labels_behavior(labels_behavior == i_behavior) = 1;
            labels_behavior_interest = find(labels_behavior == 1);
            labels_behavior_interest = labels_behavior_interest';
            
            % find bouts of current behavior
            bouts = mat2cell(labels_behavior_interest', 1, diff([0, find(diff(labels_behavior_interest) ~= 1)', size(labels_behavior_interest,1)])); 
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % ADD 2 FRAMES BEFORE EACH EGRESS BOUT (based on VBA observation)
            if i_behavior == 2 &&  ~isempty(bouts{1})
                 for i_bout = 1:size(bouts,2)
                     if bouts{i_bout}(1,1) > 2 && ~isempty(bouts{i_bout}(1,1))
                        bouts{i_bout} = bouts{i_bout}(1,1)-2:1:bouts{i_bout}(1,end);
                     end
                 end
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            % threshold duration of behavior
            bouts(cellfun('length', bouts) <= bout_threshold(i_behavior)) = [];
            % find bout onset and offset
            bout_onset = cellfun(@(v)v(1), bouts);
            bout_offset = cellfun(@(v)v(end), bouts);

            % ensure each bout has a "bout-less" baseline
            for i_bout = size(bout_onset,2):-1:2 % start from 2 since we take care of first one later
                if bout_onset(i_bout) - bout_window < bout_offset(i_bout-1) % make sure previous bout ends before baseline for new bout begins
                    bouts(i_bout) = [];
                    bout_onset(i_bout) = [];
                    bout_offset(i_bout) = []; 
                end
            end

            % ensure first bout has sufficient baseline
            bouts(bout_onset < bout_window) = [];
            bout_offset(bout_offset < bout_window) = [];
            bout_onset(bout_onset < bout_window) = [];

            % ensure last bout has sufficient end baseline
            bouts(bout_onset + bout_window > size(I_final,2)) = [];
            bout_offset(bout_offset + bout_window > size(I_final,2)) = [];
            bout_onset(bout_onset + bout_window > size(I_final,2)) = [];

            % store bout info (frames for each bout) for later quantification
            B_bout_session_info{i_behavior} = bouts;

            % extract window of fus/behavior activity around onset
            for i_bout = 1:size(bout_onset,2)
                window_idx = bout_onset(i_bout) - bout_window + 1:bout_onset(i_bout) + bout_window; % window indexes
                
                I_bout_session{i_behavior}(:,:,i_bout) = I_final(:,window_idx); % fus data in window around bout onset
                I_bout_session_during{i_behavior}{i_bout} = I_final(:,bouts{i_bout}); % fus data during bout
                
                B_bout_session{i_behavior}(:,i_bout) = labels_behavior(window_idx); % behavior data around window
                
                P_bout_session_raw_tmp = pupil(window_idx);
                P_bout_session_raw{i_behavior}(:,i_bout) = P_bout_session_raw_tmp;
                P_bout_session_norm_tmp = P_bout_session_raw_tmp - P_bout_session_raw_tmp(19);
                P_bout_session_norm{i_behavior}(:,i_bout) =  P_bout_session_norm_tmp; % substracting frame 0 (before onset)
                P_during_norm = pupil(bouts{i_bout},:)-pupil(bouts{i_bout}(1)-1);    % substracting value before bout from values in bout
                P_bout_session_during{i_behavior}{i_bout} = P_during_norm;
                
                L_bout_session{i_behavior}(:,i_bout) = labels(window_idx);
                
                V_bout_session_raw_tmp = vba(window_idx);
                V_bout_session_raw{i_behavior}(:,i_bout) = V_bout_session_raw_tmp;
                V_bout_session_norm_tmp = V_bout_session_raw_tmp - V_bout_session_raw_tmp(19);
                V_bout_session_norm{i_behavior}(:,i_bout) =  V_bout_session_norm_tmp; % substracting frame 0 (before onset)
                
            end

        end
        
        % plot behavior bout info for the current session
        fus_plot_behavior_bout_info(new_name, I_bout_session, B_bout_session, param)

        % store session info in larger total info cells for group processing
        I_bout_total{i_session} = I_bout_session;
        I_bout_total_during{i_session} = I_bout_session_during;
        
        B_bout_total{i_session} = B_bout_session;
        B_bout_total_info{i_session} = B_bout_session_info;
        
        P_bout_total_during{i_session} = P_bout_session_during;
        P_bout_total{i_session} = P_bout_session_norm;
        
        V_bout_total_during{i_session} = V_bout_session_during;
        V_bout_total{i_session} = V_bout_session_norm;
        
        L_bout_total{i_session} = L_bout_session;

    end
    
    % load segmentation one more time since we manipulated dimensions earlier in script
    [new_seg, new_name] = fus_custom_segment_PW([], RefAtlas, 0);
    
    %% Save data
    save(results_file, 'I_bout_total', 'I_bout_total_during','P_bout_total',...
        'P_bout_total_during', 'B_bout_total', 'B_bout_total_info', 'L_bout_total',...
        'V_bout_total_during', 'V_bout_total', 'V_whole_recording', 'P_whole_recording',...
        'new_name', 'new_seg', 'mouse_names', 'session_dates');
        
%             save(results_file, 'I_bout_total', 'I_bout_total_during','P_bout_total',...
%         'P_bout_total_during', 'B_bout_total', 'B_bout_total_info', 'L_bout_total',...
%         ,'V_bout_total_during', 'V_bout_total',
%         'new_name', 'new_seg', '-v7.3');
else
    results = [];
end




