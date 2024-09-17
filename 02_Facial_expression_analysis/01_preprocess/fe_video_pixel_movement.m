function fe_video_pixel_movement(data, param)
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman



rewrite = param.movt.rewrite;

% prototype build folders per mouse
for i_mouse = 1:size(data.mouse,2)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Face analysis
    
    % face image folders
    [fold_face, fold_face_hog] = fe_identify_video_fold(data, i_mouse, 'FACE');
    
    % specify spout coord file from preprocessing (always in first folder)
    coord_file = [fold_face_hog{1} 'crop_coord.mat'];
    if exist(coord_file,'file'); load(coord_file); end
    
    % specify spout coord file from preprocessing (always in first folder)
    spout_coord_file = [fold_face_hog{1} 'spout_coord.mat'];
    if exist(spout_coord_file,'file'); load(spout_coord_file); end
    
    for i_fold = 1:size(fold_face,2)
        
        save_file = [fold_face_hog{i_fold} 'face_movement.mat'];
        if ~exist(save_file,'file') || exist(save_file,'file') && rewrite == 1
        
            % load video frames
            files = dir(fold_face{i_fold});
            filename = {files.name}';
            filename = filename(cellfun(@(s)~isempty(regexp(s,'.jpg')),filename));
            
            if ~isempty(filename) && size(filename,1) > 12500
                % load first face and spout image to initialize
                clear face_diff spout_diff spout_intensity

                face = imread([fold_face{i_fold} filename{1}]);
                face = face(cropInfo.coords(1,2):cropInfo.coords(2,2),:); 
                SP_coords = spoutInfo.coords*param.pix_per_cell;
                face(SP_coords(1,2):SP_coords(2,2), SP_coords(1,1):SP_coords(2,1)) = 0;
                face = face(1:2:end, 1:2:end);
                
                spout = imread([fold_face{i_fold} filename{1}]);
                spout = spout(cropInfo.coords(1,2):cropInfo.coords(2,2),:);
                spout = spout(SP_coords(1,2):SP_coords(2,2), SP_coords(1,1):SP_coords(2,1));
                spout = spout(1:2:end, 1:2:end);
                
                spout_intensity(1) = sum(spout(:));
                
                face_img = zeros(size(face,1), size(face,2), size(filename,1));
                spout_img = zeros(size(spout,1), size(spout,2), size(filename,1));
                
                % load all data first to speed things up
                updateWaitbar = waitbarParfor(size(filename,1),save_file);
                cropInfoCoords = cropInfo.coords;
                parfor (i_file = 1:size(filename,1),6)
                    updateWaitbar()
                    
                    face_next = imread([fold_face{i_fold} filename{i_file}]); % load next face image to compare
                    face_next = face_next(cropInfoCoords(1,2):cropInfoCoords(2,2),:); % crop face image
                    face_next(SP_coords(1,2):SP_coords(2,2), SP_coords(1,1):SP_coords(2,1)) = 0;
                    face_next = face_next(1:2:end, 1:2:end);
                    face_img(:,:,i_file) = face_next; 
                    
                    spout_next = imread([fold_face{i_fold} filename{i_file}]);
                    spout_next = spout_next(cropInfoCoords(1,2):cropInfoCoords(2,2),:);
                    spout_next = spout_next(SP_coords(1,2):SP_coords(2,2), SP_coords(1,1):SP_coords(2,1));
                    spout_next = spout_next(1:2:end, 1:2:end);
                    spout_img(:,:,i_file) = spout_next;
                    
                end
                
                % extract "difference" between frames for oralfacial movement
                face = face_img(:,:,1); spout = spout_img(:,:,1);
                for i_file = 2:size(filename,1)-1

                    face_next = face_img(:,:,i_file);
                    spout_next = spout_img(:,:,i_file);

                    face_diff_tmp = face_next - face;
                    face_diff_tmp = abs(face_diff_tmp(:)); % find face difference
                    face_diff(i_file) = sum(face_diff_tmp);
                    
                    spout_diff_tmp = spout_next - spout;
                    spout_diff_tmp = abs(spout_diff_tmp(:)); % find spout difference
                    spout_diff(i_file) = sum(spout_diff_tmp);
                    
                    spout_intensity(i_file) = sum(spout_next(:));
                    
                    face = face_next;
                    spout = spout_next;
                end
                
                save(save_file, 'face_diff', 'spout_diff', 'spout_intensity');
                fprintf('\nSaving: %s\n', save_file)
                clear face_img spout_img
            end
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Body analysis
    
    % body image folders
    [fold_body, fold_body_hog] = fe_identify_video_fold(data, i_mouse, 'BODY');    
    
    body_coord_file = [fold_body_hog{1} 'body_crop_coord.mat'];
    if exist(body_coord_file,'file')
        load(body_coord_file)
    else
        cropInfo = fe_findCropCoords(fold_body{1},fold_body_hog{1}, param);
        fprintf('\nSaving: %s\n', body_coord_file);
        save(body_coord_file, 'cropInfo')
    end
    body_coords = cropInfo.coords;

    wheel_coord_file = [fold_body_hog{1} 'wheel_coord.mat'];
    if exist(wheel_coord_file,'file')
        load(wheel_coord_file)
    else
        cropInfo = fe_findCropCoords(fold_body{1},fold_body_hog{1}, param);
        fprintf('\nSaving: %s\n', wheel_coord_file);
        save(wheel_coord_file, 'cropInfo')
    end
    wheel_coords = cropInfo.coords;
    
    % extract "difference" between frames for body movement
    for i_fold = 1:size(fold_body,2)
        
        save_file = [fold_body_hog{i_fold} 'body_movement.mat'];
        if ~exist(save_file,'file') || exist(save_file,'file') && rewrite == 1
        
            % load video frames
            files = dir(fold_body{i_fold});
            filename = {files.name}';
            filename = filename(cellfun(@(s)~isempty(regexp(s,'.jpg')),filename));
            
            if ~isempty(filename) && size(filename,1) > 12500
                % load first body and wheel image to initialize
                clear body_diff wheel diff
                
                body = imread([fold_body{i_fold} filename{1}]);
                body = body(body_coords(1,2):body_coords(2,2),:);
                body = body(1:2:end,1:2:end);

                wheel = imread([fold_body{i_fold} filename{1}]);
                wheel = wheel(wheel_coords(1,2):wheel_coords(2,2),:);
                wheel = wheel(1:2:end,1:2:end);
                
                body_img = zeros(size(body,1), size(body,2), size(filename,1));
                wheel_img = zeros(size(wheel,1), size(wheel,2), size(filename,1));
                
                % load all data first to speed things up
                updateWaitbar = waitbarParfor(size(filename,1),save_file);
                parfor (i_file = 1:size(filename,1),6)
                    updateWaitbar()
                    
                    body_next = imread([fold_body{i_fold} filename{i_file}]); % load next body image to compare
                    body_next = body_next(body_coords(1,2):body_coords(2,2),:);
                    body_next = body_next(1:2:end, 1:2:end);
                    body_img(:,:,i_file) = body_next;
                    
                    wheel_next = imread([fold_body{i_fold} filename{i_file}]);
                    wheel_next = wheel_next(wheel_coords(1,2):wheel_coords(2,2),:);
                    wheel_next = wheel_next(1:2:end, 1:2:end);
                    wheel_img(:,:,i_file) = wheel_next;
                    
                end
                
                % extract "difference" between frames for oralfacial movement
                body = body_img(:,:,1); wheel = wheel_img(:,:,1);
                for i_file = 2:size(filename,1)-1

                    body_next = body_img(:,:,i_file);
                    wheel_next = wheel_img(:,:,i_file);

                    body_diff_tmp = body_next - body;
                    body_diff_tmp = abs(body_diff_tmp(:)); % find body difference
                    body_diff(i_file) = sum(body_diff_tmp);
                    
                    wheel_diff_tmp = wheel_next - wheel;
                    wheel_diff_tmp = abs(wheel_diff_tmp(:)); % find wheel difference
                    wheel_diff(i_file) = sum(wheel_diff_tmp);
                    
                    body = body_next;
                    wheel = wheel_next;
                end

                save(save_file, 'body_diff', 'wheel_diff');
                fprintf('\nSaving: %s\n', save_file)
                
                clear wheel_img body_img
            end
        end
    end
    
end
        
        
        
        
        