function video_files = create_corrected_video_brad(video_folder, camera_folder, video_jpgs, fs, rewrite)


% max video length
dur_thresh = 30; % min
dur_total = size(video_jpgs,1)/fs/60; % min

video_files = cell(0);
for i_vid = 1:ceil(dur_total/dur_thresh)

    % create "sub-video" for current run
    prts = strsplit(video_folder,'\');
    prts_cam = strsplit(camera_folder,'\');
    video_files{i_vid} = [video_folder '\' prts{end} '_' prts_cam{3} '_' prts_cam{end-1} '_' prts_cam{5} '_' num2str(i_vid) '.avi'];
    % define frames for current sub video
    frames = 1 + dur_thresh*60*fs*(i_vid - 1):dur_thresh*60*fs +dur_thresh*60*fs*(i_vid - 1);
    frames_cut = find(frames > size(video_jpgs,1));
    frames(frames_cut) = [];
    
    if ~exist(video_files{i_vid},'file') || exist(video_files{i_vid},'file') && rewrite == 1
    
        close all
        aviObj = VideoWriter(video_files{i_vid});
        set(aviObj,'FrameRate',fs);
        open(aviObj);
    
        close all
        f = figure(1);
        set(gcf,'color','w');
        axis off
        
        for i_frame = 1:size(frames,2)
    
            current_frame = imread([camera_folder video_jpgs{frames(i_frame)}]);
            % check size (should be a "nice" number)
            sz = size(current_frame); sz = round(sz/5)*5;
            % pad if necessary
            current_frame = [zeros(sz(1) - size(current_frame,1),size(current_frame,2)); current_frame];
            current_frame = [zeros(size(current_frame,1), sz(2) - size(current_frame,2)) current_frame];
    
            imshow(current_frame);
            drawnow
    
            frame = getframe(gca);
            frame.cdata = mat2gray(frame.cdata);
            
            % get size of first frame for standardization
            if i_frame == 1; SZ = size(frame.cdata); end
            % ensure all frames the same size for saving
            if ~isequal(size(frame.cdata), SZ)
                frame.cdata = rescale(imresize(frame.cdata,SZ(1:2)));
            end
            
            writeVideo(aviObj,frame);
        end
    
        close(aviObj);
        close(f)
    
        fprintf('Corrected video Created: %s\n', video_files{i_vid})
    end
end