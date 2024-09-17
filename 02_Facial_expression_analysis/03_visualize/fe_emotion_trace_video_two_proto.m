% function fe_emotion_trace_video(hog_file, trial_num, time)
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman


% trial #1
hog_file = 'J:\Bradley Edelman\Facial_Exp_fUS_HOGS\20210930\mouse_191_2\peanut_oil\FACE\Proto_corr.mat';

fs = 20;
trial_num = 1;
before = 4*fs;
after = 30*fs;

first_stim = 3600; stim_interval = 2440;
stim_start = first_stim + stim_interval*(trial_num-1);
start_idx = stim_start - before;
end_idx = stim_start + after;

maxVL = 20*60*fs;
if start_idx < maxVL && end_idx > maxVL
    error('time window spans two videos')
elseif start_idx < maxVL && end_idx < maxVL
    vid_num = '1';
elseif start_idx > maxVL && end_idx > maxVL
    vid_num = '2';
end

if exist(hog_file,'file')
    load(hog_file)
end

% Identify and load corresponding video file
parts = strsplit(hog_file, '\');

dataset = {'CUP','3PIN','FC','Odor_test'};
for i = 1:size(dataset,2)
    video_file = [parts{1} '\' parts{2}  '\Facial_Exp_Movies\'...
        dataset{i} '\FACE\' parts{5} '_' parts{4} '_' parts{6} '_FACE_' num2str(vid_num) '.avi'];
        
    if exist(video_file,'file')
        v = VideoReader(video_file);
        return
    end
end
fr = read(v,1);
%%
% automate video file
video_file = ['J:\Bradley Edelman\Facial_Exp_fUS_HOGS\HOG_videos\' parts{5} '_' parts{4} '_' parts{6} '_HOG_video.avi'];

% Initialize video format
f = figure(100); clf
subplot(3,1,1:2);
imshow(fr+25);
subplot(3,1,3)
cla; hold on

% specify prototypes, colors and labels
CC_plot = [54 130 58; 141 29 89]; % green, purple ([172 52 52] - red);
label = {'pleasure','disgust'}; 
proto = [CC(3,:); CC(2,:)]; % pleasure, disgust
proto = proto(:,start_idx:end_idx);
% proto = exp(proto);

if size(proto,1) > 1 % normalize prototypes if more than one
    for i = 1:size(proto,1)
        proto(i,:) = proto(i,:) - mean(proto(i,1:1 + before));
    end
end
proto(proto <= 0) = 0;
% figure(10); clf; plot(proto')
    
% Plot trace
for i_proto = 1:size(proto,1)
    tr{i_proto} = plot(-before:after,nan(1,size(proto,2)),'color',CC_plot(i_proto,:)/255,'linewidth',1.5);
end

% stimulation
rectangle('position',[0 max(proto(:))+0.025 fs*2 0.0125],'facecolor','b');

% adjust axes
set(gca,'xlim',[-before after],'ylim',[min(proto(:)) - 0.025 max(proto(:)) + 0.05],'ytick','')
set(gca,'xtick',-before:fs*4:after,...
    'xticklabel',strsplit(num2str([-before:fs*4:after]/fs)))
set(gcf,'color','white')
ylabel('Proto. Corr.'); xlabel('time (sec)');
% title(E)


aviObj = VideoWriter(video_file);
set(aviObj,'FrameRate',20);
open(aviObj);


video_frames = start_idx:end_idx;
for i_frame = 1:size(video_frames,2)
    
    subplot(3,1,1:2); cla
    fr = read(v,video_frames(i_frame)); % need to extract same frame
    imshow(fr+25);
    
    for i_proto = 1:size(proto,1)
        tr{i_proto}.YData(i_frame) = proto(i_proto,i_frame);
    end
    drawnow
    pause(0.1)
    
    frame = getframe(f);
    writeVideo(aviObj,frame);
    
end
    
close(aviObj);
close(f)

fprintf('\nCreating: %s\n', video_file);

