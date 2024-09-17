function fe_video_clip(video_file, labeled_video_file, d, time, d_thresh, idx)
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman


% labeled_video_file = 'J:\Bradley Edelman\DeepLabCuts\CUPx_face-brad-2021-05-26\videos\mouse_0070_20210411_quinine_FACEDLC_resnet_50_CUPx_faceMay26shuffle1_200000_filtered_labeled.mp4';
v = VideoReader(labeled_video_file);
frames = read(v,idx);

f = figure(100); clf
subplot(3,1,1:2);
imshow(frames(:,:,:,1));
subplot(3,1,3)
cla; hold on
flag = find(d(:,2) < d_thresh);
plot([time(flag)' time(flag)'],[0 1500],'color',[.8 .8 .8])
slide = plot([time(1) time(1)],[0 1500],'color','k','linewidth', 2);
scatter(time, d(:,2), 5, C(2,:));
set(gca,'ylim',[0 1500],'xlim',[0 max(time)])
ylabel('distance (pixels)'); xlabel('time (sec)');
title('Distance: Clean')


aviObj = VideoWriter(video_file);
set(aviObj,'FrameRate',40);
open(aviObj);

C = [0 0 1; 1 0 .5; .6 .6 0];
for i_frame = idx(1):idx(2)
    subplot(3,1,1:2)
    imshow(frames(:,:,:,i_frame));
    
    
    set(slide,'xdata',[time(i_frame) time(i_frame)])
    if ismember(i_frame,flag)
        set(slide,'color','r')
    else
        set(slide,'color','k')
    end
    
    frame = getframe(f);
    writeVideo(aviObj,frame);

end

close(aviObj);
close all
