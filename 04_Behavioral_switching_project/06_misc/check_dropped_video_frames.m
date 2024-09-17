function [n, video_names_sort, flag_correct] = check_dropped_video_frames(video_folder, fs)

% load videography info
video_data = dir([video_folder '\*.jpg']);
video_names = {video_data.name}';
video_names = video_names(3:end);

if isempty(video_names)
    
    n = [];
    video_names_sort = [];
    flag_correct = -1;
    fprintf('\nNo video jpg frames detected in %s \n', video_folder)

else

    % extract numbers of frames
    for i_fr = 1:size(video_names,1)
        prts = strsplit(video_names{i_fr},'-');
        order(i_fr) = str2double(prts{3}(1:end-4));
    end
    [B,I] = sort(order,'ascend');
    video_names_sort = video_names(I);
    
    % check for dropped frames
    for i_fr = 1:size(video_names_sort,1)
        prts = strsplit(video_names_sort{i_fr},'-');
        cnt{i_fr} = prts{2}(end-5:end);
    
        cnt_hr(i_fr) = str2double(cnt{i_fr}(1:2));
        cnt_min(i_fr) = str2double(cnt{i_fr}(3:4));
        cnt_sec(i_fr) = str2double(cnt{i_fr}(5:6));
    end
    
    % count how many frames actually acquired each second
    unNum   = unique(str2double(cnt(:)),'stable');
    [n,bin] = histc(str2double(cnt(:)),unNum);
    
    
    total_sec = size(n,1);
    if n(1) + n(end) <= fs; total_sec = total_sec - 1; end
    frames_total = total_sec*fs;
    frames_true = size(video_names_sort,1);
    
    if frames_true + 2*fs >= frames_total % dropped less than 2 sec
        flag_correct = 0; % no correction, create video
    elseif frames_true + 0.1*frames_total < frames_total % dropped more than 10% of frames
        flag_correct = -1; % too much data loss, no correction, no video
    else
        flag_correct = 1; % salvagable video?, yes correction, yes video
    end
    
    fprintf('\nChecking for dropped frames in %s \n', video_folder)

end



