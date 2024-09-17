function fe_check_emotion_prototype_folders(data, param)
% check for each mouse that the prototype folder index "matches" the emotion 

% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman


% prototype build folders per mouse
for i_mouse = 1:size(data.mouse,2)
    
    fprintf('\n\nMouse #: %s', data.mouse{i_mouse})
    
    [fold_face, fold_face_hog] = fe_identify_video_fold(data, i_mouse, 'FACE');
    
    for i_fold = 1:size(data.idxProto,2)
        
        if ~isequal(data.idxProto(i_mouse,i_fold),0)
        
            % also create folder containing information for each prototype
            iparts = strsplit(fold_face_hog{data.idxProto(i_mouse,i_fold)},'\');
            
            % print out the stimulus folder type being used to create each proto
            if ismember(data.idProto{i_fold}, {'tail_shock', 'freeze', 'escape'})
                if ~strcmp(iparts{7}, {'tail_shock'})
                    cprintf('red', '\nProto: %s, Stim: %s, Date: %s', data.idProto{i_fold}, iparts{7}, iparts{5});
                else
                    fprintf('\nProto: %s, Stim: %s, Date: %s', data.idProto{i_fold}, iparts{7}, iparts{5})
                end
            else
                if ~strcmp(iparts{7}, data.idProto{i_fold})
                    cprintf('red', '\nProto: %s, Stim: %s, Date: %s', data.idProto{i_fold}, iparts{7}, iparts{5});
                else
                    fprintf('\nProto: %s, Stim: %s, Date: %s', data.idProto{i_fold}, iparts{7}, iparts{5})
                end
            end
            
        end
    end
end