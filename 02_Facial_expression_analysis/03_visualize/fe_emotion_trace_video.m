% function fe_emotion_trace_video(hog_file, trial_num, time)
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman


% trial #2
hog_file = 'J:\Bradley Edelman\Facial_Exp_fUS_HOGS\20210524\mouse_0127\sucrose\FACE\Proto_corr.mat';
% trial #1
hog_file = 'J:\Bradley Edelman\Facial_Exp_fUS_HOGS\20210524\mouse_0127\tail_shock\FACE\Proto_corr.mat';
time = [80 400];
trial_num = 2;

first_stim = 3600; stim_interval = 2440; num_stim = 7; Lstim = 40; Lbase = 80;
[idxStim, idxBase, idxExtend] = fe_idxExtract(first_stim, stim_interval, num_stim, Lstim, Lbase);

if exist(hog_file,'file')
    load(hog_file)
end

% Identify and load corresponding video file
parts = strsplit(hog_file, '\');
stim = parts{6};

dataset = {'CUP','3PIN'};
for i = 1:size(dataset,2)
    video_file = [parts{1} '\' parts{2} '\' parts{3}(1:end-5) '\Facial_Exp_Movies\'...
        dataset{i} '\FACE\' parts{5} '_' parts{4} '_' parts{6} '_FACE.avi'];
        
    if exist(video_file,'file')
        v = VideoReader(video_file);
        return
    end
end
fr = read(v,1);

% automate video file
video_file = ['J:\Bradley Edelman\Facial_Exp_fUS_HOGS\HOG_videos\' parts{5} '_' parts{4} '_' parts{6} '_HOG_video.avi'];

% Initialize video format
f = figure(100); clf
subplot(3,1,1:2);
imshow(fr+25);
subplot(3,1,3)
cla; hold on

switch stim
    case 'sucrose'
        CC_plot = [54 130 58];
        proto = CC(3,:);
        E = 'Pleasure';
    case 'quinine'
        CC_plot = [141 29 89];
        proto = CC(2,:);
        E = 'Disgust';
    case 'tail_shock'
        CC_plot = [172 52 52];
        proto = CC(4,:);
        stim = 'tail shock';
        E = 'Pain';
end
% keep only "extended" data from each trial
proto = proto(idxExtend);
proto = reshape(proto,[],num_stim);
idxExtend = reshape(idxExtend,[],num_stim);

% specify time of video and adjust field of view
proto = proto(:,trial_num);
idxExtend = idxExtend(:,trial_num);

% Plot trace
tr = plot(1:size(proto,1),nan(1,size(proto,1)),'color',CC_plot/255,'linewidth',1.5);

% stimulation
rectangle('position',[2*Lbase max(proto)+0.025 Lstim 0.0125],'facecolor','b');

% adjust axes
set(gca,'xlim',[time(1) time(2)],'ylim',[min(proto) - 0.025 max(proto) + 0.05],'ytick','')
set(gca,'xtick',0:Lbase:2*Lbase+10*Lstim,...
    'xticklabel',strsplit(num2str([-2*Lbase:Lbase:2*Lbase+10*Lstim]/20)))
set(gcf,'color','white')
ylabel('Proto. Corr.'); xlabel('time (sec)');
title(E)


aviObj = VideoWriter(video_file);
set(aviObj,'FrameRate',20);
open(aviObj);

for i_frame = time(1):time(2)
    
    subplot(3,1,1:2); cla
    fr = read(v,idxExtend(i_frame)); % need to extract same frame
    imshow(fr+25);
    
    tr.YData(i_frame) = proto(i_frame);
    drawnow
    
    frame = getframe(f);
    writeVideo(aviObj,frame);
    
end
    
close(aviObj);
close(f)

fprintf('\nCreating: %s\n', video_file);

