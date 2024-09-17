addpath(genpath('C:\Users\bedelman\Documents\GitHub\MaceGogolla_lab\paulina\package\06_misc\'))

data_folder = 'D:\BRAD_FACIAL_EXP\20230421\'; 
vid_folder = '\\nas6\datastore_brad$\Facial_Exp_Movies\New_stimuli_test\';

if ~isfolder(vid_folder); mkdir(vid_folder); end

fs = 20; % video sampling rate

% find mice options in data folder
mouse_options = dir(data_folder);
for i_mouse = 3:size(mouse_options,1)
    
    mouse_folder = [data_folder mouse_options(i_mouse).name '\'];

    stim_options = dir(mouse_folder);
    for i_stim = 3:size(stim_options,1)
        stim_folder = [mouse_folder stim_options(i_stim).name '\'];
    
        % find camera options in mouse folder
        camera_options = dir(stim_folder);
        for i_cam = 3:size(camera_options,1)
        
            camera_folder = [stim_folder camera_options(i_cam).name '\'];
            
            % define
            prts = strsplit(camera_folder, '\');
            mouse_tmp = find(cellfun(@(v) ~isempty(v), strfind(prts,'mouse'))); % change to "PW" strfind(prts,'PWWT')))
%             video_folder = [vid_folder prts{mouse_tmp} '_' prts{mouse_tmp + 1}];
            video_folder = [vid_folder prts{mouse_tmp}];
            
            % takes a while so make sure we need to do it...
%             if ~isfolder(video_folder) || size(dir(video_folder),1) <= 2
                
                mkdir(video_folder)
                
                % check for dropped video frames
                [n, video_jpgs, flag_correct] = check_dropped_video_frames(camera_folder, fs);
    
                % correct dropped frames in video
                video_jpgs = correct_dropped_frames2(fs, n, video_jpgs, flag_correct);
    
                % create video from corrected jpgs
                rewrite = 0;
                video_files = create_corrected_video_brad(video_folder, camera_folder, video_jpgs, fs, rewrite);
                
                % delete folder of jpgs
    %             rmdir(camera_folder)
            
%             end
        end
    end
end






