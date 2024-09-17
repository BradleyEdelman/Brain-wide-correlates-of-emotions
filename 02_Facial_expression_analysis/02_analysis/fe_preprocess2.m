function fe_preprocess2(data, param)
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman


rewrite = param.hog_preproc.rewrite;

for i_mouse = 1:size(data.mouse,2)
    
    % specify image folders and corresponding hog folders to save to
    [fold, fold_hog] = fe_identify_video_fold(data, i_mouse, video_type);
    
    % convert images in each folder to hogs
    for i_fold = 1:size(fold,2)
        
        % specify hog file
        hog_file = [fold_hog{i_fold} 'hogs.mat'];
        coord_file = [fold_hog{1} 'crop_coord.mat'];
        spout_coord_file = [fold_hog{1} 'spout_coord.mat'];
        
        if ~exist(hog_file,'file') || rewrite == 1
            
            % for first folder specify/create crop coordinates
            if i_fold == 1
                
                if exist(coord_file,'file')
                    load(coord_file)
                else
                    cropInfo = fe_findCropCoords(fold{1},fold_hog{1}, param);
                    fprintf('\nSaving: %s\n', coord_file);
                    save(coord_file, 'cropInfo')
                end
                
                % crop and create hogs (no registration needed for first folder)
                [hog, filename] = fe_imagesToHogsCellCrop(fold{1}, param, cropInfo);
                
                if ~exist(spout_coord_file,'file')
                    spoutInfo = fe_findSpoutCoords(fold{1}, fold_hog{1}, param, cropInfo);
                    fprintf('\nSaving: %s\n', spout_coord_file);
                    save(spout_coord_file, 'spoutInfo')
                end
                
            elseif i_fold > 1 && exist(coord_file,'file')
                
                if ~exist('coordInfo','var'); load(coord_file); end
                
                % register (to first folder), crop and create hogs
                hog = fe_imagesToHogsCellCropAlign(fold{i_fold}, fold{1}, param, cropInfo, 'saveFolder', fold_hog{i_fold});
                
            end
            
            if ~isempty(hog)
                fprintf('\nSaving: %s\n', hog_file);
                save(hog_file, 'hog', '-v7.3')
            else
                fprintf('\nNOT Saving: %s\n', hog_file);
            end
            
        else
            fprintf('\nAlready exists: %s\n', hog_file);
        end
        
    end
end
    
    
    
