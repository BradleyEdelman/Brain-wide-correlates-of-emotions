%% Converting behavioral bout videos to jpg folders

% Mace Lab - Max Planck Institute of Neurobiology
% Author: Bradley Edelman
% Date: 05.01.22


video_fold = 'J:\Paulina Gabriele Wanken\Data\3D\01_behavior_fus_analysis\behavior_bout_videos_1\groom\';
jpg_fold = 'J:\Paulina Gabriele Wanken\Data\3D\01_behavior_fus_analysis\behavior_bout_videos_1_jpg\groom\';
% jpg_fold = 'C:\Users\bedelman\Desktop\';
if ~isdir(jpg_fold); mkdir(jpg_fold); end 

videos = dir([video_fold '*.avi']);
for i_vid = 1:size(videos,1)

    current_video = [video_fold videos(i_vid).name];
    current_jpg = [jpg_fold videos(i_vid).name(1:end-4) '\'];
    if ~isdir(current_jpg); mkdir(current_jpg); end
    
    vid = VideoReader(current_video);
    for i_fr = 1:vid.NumFrames
        frame_tmp = readFrame(vid);
        frame_tmp = rgb2gray(frame_tmp);
        
        % find padding attached during figure capture for bout video creation
        padding_x = find(int16(mean(frame_tmp,1)) > 200);
        padding_y = find(int16(mean(frame_tmp,2)) > 200);
        
        % remove padding
        frame_tmp(padding_y,:) = [];
        frame_tmp(:,padding_x) = [];
        
        figure(100); clf;  h = imagesc(frame_tmp);
        axis off; colormap gray
        
        % save frame as .jpg (without annoying matlab figure border)
        imwrite(h.CData, [current_jpg 'GROOM-' num2str(i_fr) '.jpg'], 'jpg')
    end
end
    
    
    
    






