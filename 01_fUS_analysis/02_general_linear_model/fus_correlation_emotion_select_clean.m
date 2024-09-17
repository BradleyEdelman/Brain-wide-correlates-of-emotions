function fus_correlation_emotion_select_clean(data, param)

close all

proc_load{1} = 'mask'; proc_load{2} = param.art.suffix;
proc_save = {'corr_emotion_select'};

rewrite = param.corr_emotion.rewrite;

emotion = '\\nas6\datastore_brad$\Facial_Exp_fUS_HOGS\';
for i_mouse = 1:size(data.mouse,2)
    
    for i_run = 1:size(data.mouse(i_mouse).run,2)
        
        close all
        % file and directory information
        parts = strsplit(data.mouse(i_mouse).run{i_run}, '_'); date = parts{1};
        if strfind(date,'2021') == 5; date = [date(5:end) date(1:4)]; end
        if strcmp(date(4),'0'); date(4) = '1'; end
        label = data.mouse(i_mouse).label{i_run};
        
        % correct some lableing issues here
        if strcmp(label,'tail shock'); label = 'tail_shock'; end
        
        % identify supporting files
        hog_fold = [emotion date '\mouse_' data.mouse(i_mouse).id '\' label];
        proto_file = [hog_fold '\FACE\Proto_corr_all_POST_NORMALIZATION.mat'];
        face_movt_file = [hog_fold '\FACE\face_movement.mat'];
        body_movt_file = [hog_fold '\BODY\body_movement.mat'];
        
        % storage locations
        storage = [data.raw_fold data.mouse(i_mouse).id '\fus\' data.mouse(i_mouse).run{i_run} '\'];
        save_fold = [storage char(proc_save) '\'];
        save_file = [save_fold 'I_' char(proc_save) '_' param.art.suffix param.iter.suffix param.note.suffix '.mat'];
        
        if exist(proto_file,'file') && exist(save_file,'file') && rewrite == 1 ||...
                exist(proto_file,'file') && ~exist(save_file,'file')
            
            [i_mouse i_run]
            % check if necessary files/folders exist
            proc_file = fus_check(storage, proc_load, proc_save);
            proc_file = proc_file(cellfun(@(v) ~contains(v,'NEW_FORMAT'), proc_file));
            for i = 1:size(proc_file,1)
                while ~exist('TMP','var'); TMP = load(proc_file{i}); end; load(proc_file{i}); clear TMP
            end
            
            % load emotion prototype correlations and bin into fUS temporal sampling
            protoInfo = load(proto_file);
            % define which regressors you want to keep
            select = param.corr_emotion.select;
            select(strcmp('movt',select) | strcmp('face',select)) = [];
            
            % ensure correct "I" being used based on the artifact removal method utilized
            eval(['I_input = I_' param.art.suffix '_mean;'])
            [I_emotion, fUS_proto] = fus_proto_regressor_select(I_input, protoInfo.D_fus, protoInfo.order, select, param);
            
            reg_labels = param.corr_emotion.select;
            reg_labels = {'stim' reg_labels{:}};
            
            % reshape into voxels x time
            I_emotion1 = reshape(I_emotion, [], size(I_emotion,3));
            
            % Add body and face movement regressors if possible
            if exist(body_movt_file,'file') && ismember('movt',param.corr_emotion.select)
            	while ~exist('TMP','var'); TMP = load(body_movt_file); end; load(body_movt_file); clear TMP
                [body_out, body_out_norm] = fus_video_to_fus(body_diff, param);
                if size(body_out,2) >= size(I_emotion1,2)
                    body_out = body_out(1:size(I_emotion1,2));
                    body_out_norm = body_out_norm(1:size(I_emotion1,2));
                    fUS_proto.regressor = [fUS_proto.regressor; body_out];
                    fUS_proto.regressor_norm = [fUS_proto.regressor_norm; body_out_norm];
                else
                    label_idx = find(strcmp('movt',reg_labels));
                    reg_labels(label_idx) = [];
                end
            else
                label_idx = find(strcmp('movt',reg_labels));
                reg_labels(label_idx) = [];
            end
            
            if exist(face_movt_file,'file') && ismember('face',param.corr_emotion.select)
            	while ~exist('TMP','var'); TMP = load(face_movt_file); end; load(face_movt_file); clear TMP
                [face_out, face_out_norm] = fus_video_to_fus(face_diff, param);
                if size(face_out,2) >= size(I_emotion1,2)
                    face_out = face_out(1:size(I_emotion1,2));
                    face_out_norm = face_out_norm(1:size(I_emotion1,2));
                    fUS_proto.regressor = [fUS_proto.regressor; face_out];
                    fUS_proto.regressor_norm = [fUS_proto.regressor_norm; face_out_norm];
                else
                    label_idx = find(strcmp('face',reg_labels));
                    reg_labels(label_idx) = [];
                end
            else
                label_idx = find(strcmp('face',reg_labels));
                reg_labels(label_idx) = [];
            end
            
            % determine if each stimulus evoked a correct emotion bout
            stim_list = data.mouse(i_mouse).stim_sequence{i_run}{1};
            stim_dur = data.mouse(i_mouse).stim_sequence{i_run}{2};
            lag = data.mouse(i_mouse).stim_sequence{i_run}{3};
            
            % stimulation indices
            framestim = zeros(1,size(stim_list,2));
            for i=1:size(stim_list,2)
                framestim(i) = find(t_interp >= stim_list(i),1,'first');
            end
            
            % create box stimulus
            stim = zeros(1,size(I_emotion,3));
            stim_dur_fr = floor(stim_dur/param.dt_interp);
            stim(1,framestim) = 1;
            for i_dur = 1:stim_dur_fr - 1
                stim(1,framestim + i_dur) = 1;
            end
            
            % create hrf
            [hrf,p] = spm_hrf(dt_interp,[3 16 1 1 20 0 16]);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Standard (full) emotion maps
            
            % add emotion prototypes to stimulus regressor
            stim = stim(1:size(I_emotion1,2)); % in case video stopped recording at some point :/ (use the data that you can)
            
            % Scrub stimulus if specified
            if param.corr.scrub_stimulus == 1
                stim_full = fUS_proto.regressor_norm;
                stim_full(:,stim == 1) = [];
                reg_labels(1) = [];
            else
                stim_full = [stim; fUS_proto.regressor_norm];
            end
            
            % remove empty regressors
            empty_idx = find(sum(stim_full,2) == 0);
            stim_full(empty_idx,:) = [];
            reg_labels(empty_idx) = [];
            
            % account for lag, remove circular shifted end
            stim_full = circshift(stim_full, -lag, 2);
            stim_full = stim_full(:,1:end-lag);
            
            % account for lag and scrubbing in fUS data
            if param.corr.scrub_stimulus == 1; I_emotion1(:,stim == 1) = []; end
            I_emotion11 = I_emotion1(:,1:end-lag);
            
            % pad convolution to avoid boundary issues
            clear stim_full_hrf
            stim_full_hrf = conv2([flipud(stim_full(:,1:50)'); stim_full'; flipud(stim_full(:,end-49:end)')],hrf);
            stim_full_hrf = stim_full_hrf(51:51 + size(stim_full,2)-1,:);
            
            % center regressors
            stim_orig = stim_full_hrf(:,1);
            stim_full_hrfinterp = zscore(stim_full_hrf);
            
            clear CC
            for i_regress = 1:size(stim_full_hrfinterp,2)
                i_regress
                for i_vox = 1:size(I_emotion11,1)
                    
                    tmp = corrcoef(I_emotion11(i_vox,:), stim_full_hrfinterp(:,i_regress));
                    CC(i_vox, i_regress) = tmp(1,2);
                end
            end
            
            figure(4);clf
            for i_cc = 1:size(CC,2)
                subplot(1, size(CC,2),i_cc)
                tmp = CC(:,i_cc);
                tmp = reshape(tmp,36,[]);
                tmp = fus_2d_to_3d(tmp,4);
                imagesc(tmp)
                caxis([-.2 .2]);
                axis off
                title(reg_labels{i_cc})
            end
            drawnow
            
            fprintf('\nSaving: %s\n', save_file);
            save(save_file, 'stim', 'stim_full_hrf', 'CC', 'reg_labels')   
            
        end
        
    end
    
end


