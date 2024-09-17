function fe_create_prototypes_odorants(data, param)
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman


rewrite = param.proto.rewrite;

d_thresh = param.hog_analyze.d_thresh;

Lbase = param.hog_disp.lbase;
Lstim = param.hog_disp.lstim;
numtrial = param.hog_disp.numtrial;

% prototype build folders per mouse
for i_mouse = 1:size(data.mouse,2)
    
    % specify image folders and corresponding analysis save folders
    fold = cell(0); fold_analysis = cell(0);
    for i_date = 1:size(data.date,2)
        
        path = [data.data_fold '\' data.date{i_date} '\' data.mouse{i_mouse} '\'];
        stim = dir(path); stim = {stim.name}; stim = stim(3:end);
        
        % create separate analysis folders
        path_analysis = [data.hog_fold data.date{i_date} '\' data.mouse{i_mouse} '\'];
        for i_stim = 1:size(stim,2)
            fold{end + 1} = [path stim{i_stim} '\FACE\'];
            fold_analysis{end + 1} = [path_analysis stim{i_stim} '\FACE\'];
        end
    end
    
    % specify spout coord file from preprocessing
    spout_coord_file = [fold_analysis{1} 'spout_coord.mat'];
    if exist(spout_coord_file,'file')
        load(spout_coord_file)
    end
    
    % Create protoypes
    proto_fold = [data.hog_fold  data.mouse{i_mouse} '\'];
    if ~isfolder(proto_fold); mkdir(proto_fold); end
    for i_fold = 1:size(fold,2)
        
        [idxStim, idxBase, idxExtend] = fe_idxExtract(3600, 2440, numtrial, Lstim, Lbase);
        
        iparts = strsplit(fold_analysis{i_fold},'\');
        save_file = [proto_fold 'PROTO_' iparts{end-2} '_2021112.mat'];
        if ~exist(save_file,'file') || rewrite == 1
        
            % load hogs
            hog_file = [fold_analysis{i_fold} 'hogs.mat'];
            hog = fe_hogLoadAndSpoutRemoval(hog_file, param, spoutInfo);

            % load DLC to determine if frames should be rejected
            d_thresh2 = d_thresh(1);
            if strcmp(iparts{end-2},'tail_shock'); d_thresh2 = d_thresh(2); end
            DLC_file = [fold_analysis{i_fold} 'invalid_indices_2paw.mat'];
            if exist(DLC_file,'file')
                
                load(DLC_file);
                
                idxGood = find(c == 1);
                idxBad = find(c == -2);
                
                flag1 = [];
                for i_paw = 1:2
                    flag1 = [flag1; find(d(:,end,i_paw) < d_thresh2)]; % low distance threshold
                end
                flag1 = unique(flag1);
                idxBad = unique([idxBad; flag1]);
                
            else
                
                idxGood = 1:size(hog,1);
                idxBad = [];
                
            end
            idxGood = idxGood(:); idxBad = idxBad(:);
            
            % neutral prototype first 
            save_filen = [proto_fold 'PROTO_neutral_2021112.mat'];
            if i_fold == 1
                
                % Create neutral prototype from clean frames before any stimulus given
                idxGood_neutral = idxGood;
                idxGood_neutral(idxGood_neutral > 3000) = []; 
                idxGood_neutral = idxGood_neutral(randperm(size(idxGood_neutral,1),50));
                proto_neutral = mean(hog(idxGood_neutral,:),1);
                fprintf('\nSaving: %s\n', save_filen);
                save(save_filen, 'proto_neutral')
                
            end
            load(save_filen)
            
            % correlate with neutral prototype
            CC = zeros(1, size(hog,1));
            for i_hog = 1:size(hog,1)
                i_hog
                ctmp = corrcoef(hog(i_hog,:), proto_neutral);
                CC(i_hog) = ctmp(1,2);
            end

            % plot correlation
            time = (1:size(hog,1))/20/60;

            f = figure(12); clf
            subplot(3,1,1); hold on;
            plot([time(idxBase); time(idxBase)],[.65 1],'color',[.75 .75 .75])
            plot([time(idxStim); time(idxStim)],[.65 1],'color','c')
            plot(time,CC,'k');
            title('Corr w/ Neutral: Original')
            
            subplot(3,1,2); hold on;
            plot([time(idxBase); time(idxBase)],[.75 1],'color',[.75 .75 .75])
            plot([time(idxStim); time(idxStim)],[.75 1],'color','c')
            if~isempty(idxBad); plot([time(idxBad); time(idxBad)],[.75 1],'color','r'); end
            plot(time,CC,'k');
            title('Corr w/ Neutral: Bad frames labeled')
            
            subplot(3,1,3); hold on;
            plot([time(idxBase); time(idxBase)],[.75 1],'color',[.75 .75 .75])
            plot([time(idxStim); time(idxStim)],[.75 1],'color','c')
            CC_clean = CC; CC_clean(idxBad) = nan;
            plot(time,CC_clean,'k');
            title('Corr w/ Neutral: Bad frames removed')
            drawnow
            
            % find least correlated STIM frames
            
            % remove first trial to allow animal to adjust to and process
            % stimulus (only for prototype generation)
            idxStim = reshape(idxStim,[],numtrial);
            idxStim(:,1) = [];
            idxStim = idxStim(:)';
%             idxStim = idxStim(:,2);
            
            [B,I] = sort(CC_clean,'ascend');
            I(~ismember(I,idxStim)) = [];
            B(~ismember(I,idxStim)) = [];
            
            % look at images of least correlated frames
            files = dir(fold{i_fold});
            filename = {files.name};
            filename = filename(cellfun(@(s)~isempty(regexp(s,'.jpg')),filename));
            num_img = 20; clear img_proto img_neut
            for i_fr = 1:num_img
                img_proto(:,:,i_fr) = imread([fold{i_fold} filename{I(i_fr)}]);
                img_neut(:,:,i_fr) = imread([fold{i_fold} filename{I(end-num_img+i_fr)}]);
            end
            ff = figure(213+i_fold); clf
            subplot(1,2,1); imagesc(mean(img_neut,3)); title('neutral'); axis off
            subplot(1,2,2); imagesc(mean(img_proto,3)); title('proto'); axis off
            colormap gray
            drawnow
            savefig(ff, [proto_fold iparts{end-2} '_ave_proto_img.fig'])
            saveas(ff, [proto_fold iparts{end-2} '_ave_proto_img.png'])
            
            % create proto
            proto = mean(hog(I(1:num_img),:),1);

            savefig(f, [proto_fold 'Neutral_corr_' iparts{end-2} '.fig'])
            saveas(f, [proto_fold 'Neutral_corr_' iparts{end-2} '.png'])

            fprintf('\nSaving: %s\n', save_file);
            save(save_file, 'proto')
            
        end
    end
end  