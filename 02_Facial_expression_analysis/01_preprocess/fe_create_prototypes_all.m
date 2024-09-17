function fe_create_prototypes_all(data, param)
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman


rewrite = param.proto.rewrite;

Lbase = param.hog_disp.lbase;
Lstim = param.hog_disp.lstim;
numtrial = param.hog_disp.numtrial;

% prototype build folders per mouse
for i_mouse = 1:size(data.mouse,2)
    
    fprintf('\n\nMouse #: %s', data.mouse{i_mouse})
    
    % specify image folders and corresponding hog folders to save to
    [fold_face, fold_face_hog] = fe_identify_video_fold(data, i_mouse, 'FACE');
    [fold_body, fold_body_hog] = fe_identify_video_fold(data, i_mouse, 'BODY');
    
    % specify spout coord file from preprocessing (always in first folder)
    spout_coord_file = [fold_face_hog{1} 'spout_coord.mat'];
    if exist(spout_coord_file,'file'); load(spout_coord_file); end
    
    % create folder for each mouse that contains prototype information
    fold_mouse = [data.hog_fold  data.mouse{i_mouse} '\'];
    if ~isfolder(fold_mouse); mkdir(fold_mouse); end
    
    for i_fold = 1:size(data.idxProto,2)
        
        if ~isequal(data.idxProto(i_mouse,i_fold),0)
        
            [idxStim, idxBase, idxExtend] = fe_idxExtract(3600, 2440, numtrial, Lstim, Lbase);

            % also create folder containing information for each prototype
            iparts = strsplit(fold_face_hog{data.idxProto(i_mouse,i_fold)},'\');
            
            % print out the stimulus folder type being used to create each proto
            fprintf('\nProto: %s, Stim: %s', data.idProto{i_fold}, iparts{end - 2})
            
            proto_fold = [fold_mouse data.idProto{i_fold} '\'];
            if ~isfolder(proto_fold); mkdir(proto_fold); end

            save_file = [proto_fold 'PROTO_' data.idProto{i_fold} '.mat'];
            if ~exist(save_file,'file') || rewrite == 1
                
                % load hogs
                hog_file = [fold_face_hog{data.idxProto(i_mouse,i_fold)} 'hogs.mat'];
                hog = fe_hogLoadAndSpoutRemoval(hog_file, param, spoutInfo);

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % neutral prototype first
                neutral_fold = [fold_mouse 'neutral\'];
                if ~isfolder(neutral_fold); mkdir(neutral_fold); end
                save_filen = [neutral_fold 'PROTO_neutral.mat'];
                if i_fold == 1

                    % find frames with lowest oralfacial movement
                    
                    % load face movement file
                    face_file = [fold_face_hog{data.idxProto(i_mouse,i_fold)} 'face_movement.mat'];
                    if exist(face_file,'file'); load(face_file); end
                    
                    face_diff_tmp = face_diff(1:3000);
                    [B,I] = sort(face_diff_tmp,'ascend');
                    low_idx = I(1:250);
                    proto_neutral = mean(hog(I(1:250),:),1);

                    % create and save an image of the neutral face
                    files = dir(fold_face{data.idxProto(i_mouse,i_fold)});
                    filename = {files.name}; filename = filename(cellfun(@(s)~isempty(regexp(s,'.jpg')),filename));
                    for i_fr = 1:size(low_idx,2)
                        img_neut(:,:,i_fr) = imread([fold_face{data.idxProto(i_mouse,i_fold)} filename{low_idx(i_fr)}]);
                    end

                    fprintf('\nSaving: %s\n', save_filen);
                    thresh = struct('escape', [], 'freeze', [], 'movt', [], 'face', []);
                    save(save_filen, 'proto_neutral', 'img_neut', 'thresh')

                end
                load(save_filen)

                % correlate current set of hogs with neutral prototype
                CC = zeros(1, size(hog,1));
                for i_hog = 1:size(hog,1)
                    ctmp = corrcoef(hog(i_hog,:), proto_neutral);
                    CC(i_hog) = ctmp(1,2);
                end

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % find least correlated STIMULUS frames for prototype creation
                
                % first define "stimulus" frames for elicited and spontaneous emotions
                thresh = struct('escape', [], 'freeze', [], 'movt', [], 'face', []);
                if strcmp(data.idProto{i_fold}, 'escape')
                    
                    % body movement file (extract wheel data)
                    body_file = [fold_body_hog{data.idxProto(i_mouse,i_fold)} 'body_movement.mat'];
                    
                    % define escape as rapid movement (in "fearful" context) lasting >2s (Dolensek 2020)
                    thresh.escape = 2*20; % 2 sec at 20 Hz
                    [stim, base, extend, thresh] = fe_find_escape_bouts(body_file, thresh, param);            
                    
                    idxStim = cat(2, stim{:});
                    idxBase = cat(2, base{:});
                    
                elseif strcmp(data.idProto{i_fold}, 'freeze')
                    
                    % body movement file (extract wheel data)
                    body_file = [fold_body_hog{data.idxProto(i_mouse,i_fold)} 'body_movement.mat'];
                    
                    % face movement file
                    face_file = [fold_face_hog{data.idxProto(i_mouse,i_fold)} 'face_movement.mat'];
                    
                    % define freezing as immobility (in "fearful" context) lasting >5s and "no" orofacial movement (Dolensek 2020)
                    thresh.freeze = 5*20; % 5 sec at 20 Hz
                    [stim, base, extend, thresh] = fe_find_freeze_bouts(body_file, face_file, thresh, param);
                    
                    idxStim = cat(2, stim{:});
                    idxBase = cat(2, base{:});
                    
                else % all other deliverer stimuli (sucrose, quinine, tailshock)
                    
                    % remove first trial to allow animal to adjust to and process stimulus
                    idxStim = reshape(idxStim,[],numtrial);
                    idxStim(:,1) = [];
                    idxStim = idxStim(:)';
                    
                end

                % rank least correlated indices
                [B,I] = sort(CC,'ascend');
                B(~ismember(I,idxStim)) = [];
                I(~ismember(I,idxStim)) = [];

                % create and save an image of the prototype face
                files = dir(fold_face{data.idxProto(i_mouse,i_fold)});
                filename = {files.name}; filename = filename(cellfun(@(s)~isempty(regexp(s,'.jpg')),filename));
                num_img = 10;
                for i_fr = 1:num_img % N frames least correlated from neutral
                    img_proto(:,:,i_fr) = imread([fold_face{data.idxProto(i_mouse,i_fold)} filename{I(i_fr)}]);
                end
                
                ff = figure(213+i_fold); clf
                subplot(1,2,1); imagesc(mean(img_neut,3)); title('neutral'); axis off
                subplot(1,2,2); imagesc(mean(img_proto,3)); title('proto'); axis off; colormap gray; drawnow
                savefig(ff, [proto_fold data.idProto{i_fold} '_ave_proto_img.fig'])
                saveas(ff, [proto_fold data.idProto{i_fold} '_ave_proto_img.png'])

                % create proto
                proto = mean(hog(I(1:num_img),:),1);

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % plot correlation with oralfacial movement and selected frames
                time = (1:size(hog,1))/20/60;

                f = figure(12); clf; hold on

                        if size(face_diff,2) > size(time,2); face_diff = face_diff(1:size(time,2)); end
                        if size(spout_diff,2) > size(time,2); spout_diff = spout_diff(1:size(time,2)); end

%                 if ~isempty(idxBase)
%                     plot([time(idxBase); time(idxBase)],[0 3],'color',[.75 .75 .75]) % baseline
%                 end
                plot([time(idxStim); time(idxStim)],[0 3],'color','c') % stimulus
                plot(time(1:size(face_diff,2)), zscore(face_diff)*.1 + 1.5,'g') % face movement
                plot(time(1:size(spout_diff,2)), zscore(spout_diff)*.1 + 2.5, 'r') % spout roi
                plot(time, CC, 'k') % neutral correlation
                scatter(time(I(1:num_img)), ones(1,num_img), 35, 'b', 'filled') % selected prototype frames
                title([iparts{end-2} ' Proto: spout(red), face(green), CC(black)'])

                savefig(f, [proto_fold 'Neutral_corr_' data.idProto{i_fold} '.fig'])
                saveas(f, [proto_fold 'Neutral_corr_' data.idProto{i_fold} '.png'])

                fprintf('\nSaving: %s', save_file);
                save(save_file, 'proto', 'thresh')

            end
        end
    end
end  

