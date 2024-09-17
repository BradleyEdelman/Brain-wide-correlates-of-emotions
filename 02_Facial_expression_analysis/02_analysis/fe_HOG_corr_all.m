function fe_HOG_corr_all(data, param)
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman


rewrite = param.hog_analyze.rewrite;

% extract indices of stim for analysis and display
Lbase = param.hog_disp.lbase;
Lstim = param.hog_disp.lstim;
numtrial = param.hog_disp.numtrial;

for i_mouse = 1:size(data.mouse,2)
    
    % specify image folders and corresponding hog folders to save to
    [fold_face, fold_face_hog] = fe_identify_video_fold(data, i_mouse, 'FACE');
    [fold_body, fold_body_hog] = fe_identify_video_fold(data, i_mouse, 'BODY');
    
    % specify crop and spout coord file from preprocessing
    coord_file = [fold_face_hog{1} 'crop_coord.mat'];
    if exist(coord_file,'file')
        load(coord_file)
    end
    
    spout_coord_file = [fold_face_hog{1} 'spout_coord.mat'];
    if exist(spout_coord_file,'file')
        load(spout_coord_file)
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % load prototypes
    
    fold_mouse = [data.hog_fold  data.mouse{i_mouse} '\'];
    proto_fold = dir(fold_mouse);
    proto_fold = {proto_fold.name};
    proto_fold = proto_fold(cellfun(@(s)isempty(strfind(s,'.')),proto_fold));
    
    proto_total = []; proto_name = cell(0); proto_thresh = cell(0);
    for i_proto = 1:size(proto_fold,2)
        
        proto_file = [fold_mouse proto_fold{i_proto} '\PROTO_' proto_fold{i_proto} '.mat'];
        if exist(proto_file,'file')
            
            load(proto_file);
            % store prototype hog
            if strcmp(proto_fold{i_proto}, 'neutral')
                proto_total = [proto_total; proto_neutral];
            else
                proto_total = [proto_total; proto];
            end
            % store prototype name thresholds used to create them (freeze/escape)
            proto_name{end+1} = proto_fold{i_proto};
            proto_thresh{end+1} = thresh;
        end
        
    end
    
    % load visual inspection file to determine which datasets are acceptable
    vis_inspect_file = ['\\nas6\datastore_brad$\fUS\' data.mouse{i_mouse}(end-3:end) '\fus\visual_inspection.xlsx'];
    if exist(vis_inspect_file,'file')
        [num,txt,raw] = xlsread(vis_inspect_file);
        fus_accept = txt(2:end,1:3);
    else
        fus_accept = ones(size(fold_face,2),1);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % correlate each hog dataset against all of the prototypes
    for i_fold = 1:size(fold_face,2)
        
        save_file = [fold_face_hog{i_fold} 'Proto_corr_all.mat'];
        hog_file = [fold_face_hog{i_fold} 'hogs.mat'];
        if exist(hog_file,'file') && ~exist(save_file,'file') ||...
            exist(hog_file,'file') && rewrite == 1
            
            % load hogs and remove spout indices
            hog = fe_hogLoadAndSpoutRemoval(hog_file, param, spoutInfo);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % decide whether run is acceptable/unacceptable
            
            % find visual inspection result for current trial
            parts = strsplit(hog_file,'\');
            date = {[parts{5}(5:8) parts{5}(1:4)], [parts{5}(5:8) '2020'], parts{5}, ['2020' parts{5}(5:8)]}; % rearrange different date formats
            parts{7}(strfind(parts{7},'_')) = ' ';
            % find row in vis inspect file that contains date and stim
            if iscell(fus_accept)
                accept_idx = find(contains(fus_accept(:,1),date) & contains(fus_accept(:,2),parts{7}));
                tmp_accept = fus_accept(:,3);
                if isempty(cat(1,tmp_accept{:}))
                    accept = 'yes';
                else
                    accept = fus_accept{accept_idx,3};
                end
            else
                accept = 'yes';
            end
            
            file_accept = [fold_face_hog{i_fold} 'CLEAN_PAWS_ACCEPT.txt'];
            if exist(file_accept,'file'); delete(file_accept); end
            file_reject = [fold_face_hog{i_fold} 'DIRTY_PAWS_REJECT.txt'];
            if exist(file_reject,'file'); delete(file_reject); end
            
            % create new acceptance/rejection file
            if strcmp(accept,'yes')
                fopen(file_accept,'wt')
            elseif strcmp(accept,'no') || strcmp(accept,'maybe')
                fopen(file_reject,'wt')
            end
            fclose all
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % correlate hogs with each prototype
            time = (1:size(hog,1))/20/60;
            CC = zeros(size(proto_total,1), size(hog,1));
            for i_proto = 1:size(proto_total,1)

                parfor (i_hog = 1:size(hog,1), 6)
                    ctmp = corrcoef(hog(i_hog,:), proto_total(i_proto,:));
                    CC(i_proto,i_hog) = ctmp(1,2);
                end

            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Determine "stimulus" and baseline indices for each prototype
            for i_proto = 1:size(proto_total,1)
                if strcmp(proto_name{i_proto}, 'escape')

                    body_file = [fold_body_hog{i_fold} 'body_movement.mat'];
                    [stim, base, extend, thresh] = fe_find_escape_bouts(body_file, proto_thresh{i_proto}, param);
                    
                    % keep only up to seven bouts (same number as stimuli)
                    if size(stim,2) > numtrial
                        idx = randperm(size(stim,2));
                        idx = idx(1:7);
                    else
                        idx = 1:size(stim,2);
                    end
                    
                    if ~isempty(stim)
                        stim = stim(idx); base = base(idx); extend = extend(idx);
                        stim = cellfun(@(v) v(1:Lstim), stim, 'uniformoutput', false);
                    end

                    idxStim{i_proto} = cat(2, stim{:});
                    idxBase{i_proto} = cat(2, base{:});
                    idxExtend{i_proto} = cat(2, extend{:});

                elseif strcmp(proto_name{i_proto}, 'freeze')

                    body_file = [fold_body_hog{i_fold} 'body_movement.mat'];
                    face_file = [fold_face_hog{i_fold} 'face_movement.mat'];
                    [stim, base, extend, thresh] = fe_find_freeze_bouts(body_file, face_file, proto_thresh{i_proto}, param);
                
                    % keep only up to seven bouts (same number as stimuli)
                    if size(stim,2) > 7
                        idx = randperm(size(stim,2));
                        idx = idx(1:7);
                    else
                        idx = 1:size(stim,2);
                    end
                    
                    if ~isempty(stim)
                        stim = stim(idx); base = base(idx); extend = extend(idx);
                        stim = cellfun(@(v) v(1:Lstim), stim, 'uniformoutput', false);
                    end

                    idxStim{i_proto} = cat(2, stim{:});
                    idxBase{i_proto} = cat(2, base{:});
                    idxExtend{i_proto} = cat(2, extend{:});

                elseif strcmp(proto_name{i_proto}, 'neutral')

%                     face_file = [fold_face_hog{i_fold} 'face_movement.mat'];
%                     [stim, base, extend] = fe_find_baseline_bouts(face_file, param); 
                    
                    [stim, base, extend] = fe_find_baseline_bouts2(CC, proto_name, param);
                    
                    idxStim{i_proto} = cat(2, stim{:});
                    idxBase{i_proto} = cat(2, base{:});
                    idxExtend{i_proto} = cat(2, extend{:});
                    
                else
                    
                    [idxStim_tmp, idxBase_tmp, idxExtend_tmp] = fe_idxExtract(3600, 2440, numtrial, Lstim, Lbase);
                    
                    idxStim{i_proto} = idxStim_tmp;
                    idxBase{i_proto} = idxBase_tmp;
                    idxExtend{i_proto} = idxExtend_tmp;
                    
                end
                
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % plot prototype correlations
            f = figure(13); clf; hold on
            for i_proto = 1:size(CC,1)
                
                tmpBase = idxBase{i_proto}; tmpStim = idxStim{i_proto};
                tmpBase(tmpBase > size(time,2)) = []; tmpStim(tmpStim > size(time,2)) = [];
                
                H = [0.6 + (i_proto - 1)*.4 1 + (i_proto - 1)*.4];
                if ~isempty(tmpBase) && ~isempty(tmpStim)
                    plot([time(tmpBase); time(tmpBase)],H,'color',[.75 .75 .75]) % baseline
                    plot([time(tmpStim); time(tmpStim)],H,'color','c') % stimulus
                end
            
                CC_plot = CC(i_proto,:) + repmat(.4 * (i_proto-1), 1, size(hog,1));
                plot(time, CC_plot) % prototype correlations
            end
            set(gca,'ytick', 0.4 + 0.4 * [1:size(proto_total,1)]',...
                'yticklabel', proto_name)
            xlabel('time (min)');
            iparts = strsplit(hog_file,'\');
            title(['Mouse: ' iparts{6}, '   Date: ', iparts{5}, '   Stim: ' iparts{7}])

            savefig(f, [fold_face_hog{i_fold} 'Proto_corr_all.fig'])
            saveas(f, [fold_face_hog{i_fold} 'Proto_corr_all.png'])
            
            fprintf('\nSaving: %s\n', save_file);
            save(save_file, 'CC', 'proto_total', 'proto_name',...
                'proto_thresh', 'idxStim', 'idxBase', 'idxExtend')
            
        end
    end
    
end
        
 