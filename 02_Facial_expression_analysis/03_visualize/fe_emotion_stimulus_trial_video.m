function fe_emotion_stimulus_trial_video(data, param)
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman


if strcmp(data.location,'local')
    Data = 'C:\data\experiments\Facial_Exp\';
else
    Data = param.Data;
end

%%
mouse_idx = 1;
trial_idx = 1;
date_idx = 1;
stim_type = 1;

% specify image folders and corresponding analysis save folders
fold = cell(0); fold_analysis = cell(0);

path = [data.data_fold data.date{date_idx} '\' data.mouse{mouse_idx} '\'];
stim = dir(path); stim = {stim.name}; stim = stim(3:end);
stim = stim(cellfun(@(s)isempty(regexp(s,'PNG')),stim));

% create separate analysis folders
path_analysis = [data.hog_fold data.date{date_idx} '\' data.mouse{mouse_idx} '\'];
for i_stim = 1:size(stim,2)
    fold{end + 1} = [path stim{i_stim} '\FACE\'];
    fold_analysis{end + 1} = [path_analysis stim{i_stim} '\FACE\'];
    if ~isdir(fold_analysis{end}); mkdir(fold_analysis{end}); end
end

B1 = 5; B2 = 10; % before and after stim baseline
L1 = 2; % length of stimulus
for i_fold = 1%:size(fold,2)
    
    idx_start = param.stim.list(stim_type, trial_idx);
    idx_start = idx_start - B1*20;
    idx_end = idx_start + (L1+B2)*20;
    
    % load video frames
    files = dir(fold{i_fold});
    filename = {files.name}';
    filename = filename(cellfun(@(s)~isempty(regexp(s,'.jpg')),filename));
    filename_vid = filename(idx_start:idx_end);
    
    % place indicator on frames during stim
    C = [0 1 0]; % cs +
    name = stim{i_stim};
    indicator = stim{i_stim};

    
    video_file = [fold_analysis{i_fold} name '.avi'];
    aviObj = VideoWriter(video_file);
    set(aviObj,'FrameRate',20);
    open(aviObj);
    
    L = size(idx_start:idx_end,2); % length of video in frames
    cmap = gray(256); cmap = [cmap; C]; % adjust colormap for stim indicator
    f = figure(100); set(gcf,'color','k');
    for i_fr = 1:L
        
        img = imread([fold{i_fold} filename_vid{i_fr}]);
        img = double(img);
        cla
        
        % stim indicator on frame
        if i_fr > B1*20 && i_fr < (B1+L1)*20
            img(25:75, 50:250) = 257;
            imagesc(img)
            text(58,50,indicator)
        else
            imagesc(img)
        end
        axis off; colormap(cmap); caxis([0 256])
        drawnow
 
        frame = getframe(f);
        writeVideo(aviObj,frame);
    end
    
    close(aviObj);
    close(f)

end

fprintf('\nCreating: %s\n', video_file);