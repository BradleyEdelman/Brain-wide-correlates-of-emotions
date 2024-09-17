function [fold, fold_analysis] = fe_identify_video_fold(data, i_mouse, video_type)
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman


if ismember(video_type,{'FACE','BODY'})

    % specify image folders and corresponding analysis save folders
    fold = cell(0); fold_analysis = cell(0);
    for i_date = 1:size(data.date,2)

        path = [data.data_fold '\' data.date{i_date} '\' data.mouse{i_mouse} '\'];
        stim = dir(path); stim = {stim.name}; stim = stim(3:end);

        % create separate analysis folders
        path_analysis = [data.hog_fold data.date{i_date} '\' data.mouse{i_mouse} '\'];
        for i_stim = 1:size(stim,2)
            
            path_tmp = [path stim{i_stim} '\' video_type '\'];
            if isfolder(path_tmp)
                fold{end + 1} = path_tmp;
                fold_analysis{end + 1} = [path_analysis stim{i_stim} '\' video_type '\'];
                if ~isdir(fold_analysis{end})
                    mkdir(fold_analysis{end})
                end
            end
            
        end
        
    end

else
    
    fprintf('\ninvalid video type\n')
    
end
