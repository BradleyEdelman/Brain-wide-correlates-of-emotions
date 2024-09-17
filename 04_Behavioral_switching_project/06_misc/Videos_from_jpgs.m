addpath(genpath('R:\Work - Experiments\Codes\Paulina\fus\Brad_fus_and_behavior_analysis\package_20230201\06_misc\'))
rmpath(genpath('R:\Work - Experiments\Codes\Paulina\fus\Brad_fus_and_behavior_analysis\package_20230201\06_misc\Leafy\'))

data_folder = 'D:\videos\Paulina\04042023\'; 
vid_folder = 'D:\videos\Paulina\04042023\00AVI\';

if ~isfolder(vid_folder); mkdir(vid_folder); end

fs = 20; % video sampling rate

% find mice options in data folder
mouse_options = dir(data_folder);
for i_mouse = 3:size(mouse_options,1)
    
    mouse_folder = [data_folder mouse_options(i_mouse).name '\'];
    
    % find camera options in mouse folder
    camera_options = dir(mouse_folder);
    for i_cam = 3:size(camera_options,1)
        
        camera_folder = [mouse_folder camera_options(i_cam).name '\'];
        
        % define
        prts = strsplit(camera_folder, '\');
        mouse_tmp = find(cellfun(@(v) ~isempty(v), strfind(prts,'PWWT'))); % change to "PW" strfind(prts,'PWWT')))
        video_folder = [vid_folder prts{mouse_tmp} '_' prts{mouse_tmp + 1}];
        
        % takes a while so make sure we need to do it...
        if ~isfolder(video_folder) || size(dir(video_folder),1) <= 2
            
            mkdir(video_folder)
            
            % check for dropped video frames
            [n, video_jpgs, flag_correct] = check_dropped_video_frames(camera_folder, fs);

            % correct dropped frames in video
            video_jpgs = correct_dropped_frames2(fs, n, video_jpgs, flag_correct);

            % create video from corrected jpgs
            video_files = create_corrected_video(video_folder, camera_folder, video_jpgs, fs);
            
            % delete folder of jpgs
%             rmdir(camera_folder)
            
        end
    end
end






