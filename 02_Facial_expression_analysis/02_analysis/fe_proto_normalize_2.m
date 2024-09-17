function fe_proto_normalize_2(data, param)
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman


%%
% extract indices of stim for analysis and display
Lbase = param.hog_disp.lbase; Lstim = param.hog_disp.lstim; numtrial = param.hog_disp.numtrial;
[idxStim, idxBase, idxExtend] = fe_idxExtract(3600, 2440, numtrial, Lstim*5, Lbase);

% LOAD REFERENCE PROTOTYPE FOR EACH STIMULUS
CC_total = cell(4,3,size(data.mouse,2));
for i_mouse = 1:size(data.mouse,2)
    i_mouse
    
    % specify image folders and corresponding hog folders to save to
    [fold_face, fold_face_hog] = fe_identify_video_fold(data, i_mouse, 'FACE');
    
    % find sucrose, quinine, and tail shock runs for reference
    solicit_idx{1} = find(contains(fold_face,'quinine'));
    solicit_idx{2} = find(contains(fold_face,'sucrose'));
    solicit_idx{3} = find(contains(fold_face,'tail_shock'));
     
    % Load only reference prototype for each stimulus (i.e. pleasure for sucrose)
    for i_stim = 1:size(solicit_idx,2)
        for i_fold = 1:size(solicit_idx{i_stim},2)
        
            proto_file = [fold_face_hog{solicit_idx{i_stim}(i_fold)} 'Proto_corr.mat'];
            accept_file = [fold_face_hog{solicit_idx{i_stim}(i_fold)} 'CLEAN_PAWS_ACCEPT.txt'];
            if exist(proto_file,'file') && exist(accept_file,'file')
                
                while ~exist('TMP','var'); TMP = load(proto_file); end; load(proto_file); clear TMP
                    
                CC = CC - nanmean(CC(:,1:3600),2); % remove mean of baseline

                % prototypes should be neutral, disgust, pleasure, pain, freeze, escape
                for i_proto = 1:size(CC,1)
                    tmp = CC(i_proto,:);
%                     tmp(tmp > tmp + 5*nanstd(tmp) | tmp < tmp - 5*nanstd(tmp)) = nan; % remove large artifacts
%                     tmp = movmean(tmp,10); % slight smoothing 
                    CC_total{i_proto, i_stim, i_mouse}{end+1} = tmp;
                end
            end
        end
    end
end
%%

% ensure that prototype time course is the same for each stimulus within each mouse
% % (some recordings drop a few frames so cant stack time courses so easily)
[b, a] = butter(5,0.01/20,'high');
clear CC_tmp_uniform_length CC_total_uniform
for i_mouse = 1:size(data.mouse,2)
    
    CC_total_mouse = CC_total(:,:,i_mouse);
    empty_row = cellfun('isempty',CC_total_mouse(:,1));
    CC_total_mouse(empty_row,:) = [];
    for i_stim = 1:size(CC_total_mouse,2)
        for i_proto = 1:size(CC_total_mouse,1)
            
            CC_tmp = cat(2, CC_total_mouse{i_proto, i_stim});
            sz = cellfun(@(v)size(v,2), CC_tmp);
            sz = min(sz);
            CC_tmp = cellfun(@(v)v(1:sz), CC_tmp, 'uniformoutput', false);
            CC_tmp = cellfun(@(v)fillmissing(v,'linear'), CC_tmp, 'uniformoutput', false);
            CC_tmp = cellfun(@(v)filtfilt(b,a,v), CC_tmp, 'uniformoutput', false);
            CC_tmp = cat(1, CC_tmp{:});
            
            CC_tmp_uniform_length{i_proto, i_stim} = CC_tmp;
        end
    end
    
    CC_total_uniform(:,:,i_mouse) = CC_tmp_uniform_length;
end
CC_total = CC_total_uniform;

%%

% Extract min-max normalization values from prototype time series in which
% it was build off of
clear CC_ref
for i_mouse = 1:size(data.mouse,2)
    
    % first create reference prototype for each animal and emotion
    % reference prototypes are along the diagonal (after the first row - neutral)
    for i_proto = 1:size(CC_total,1)-1
        
        if i_proto > size(CC_total,2) % escape and freeze in the tail shock runs...
            ref_proto = nanmean(CC_total{i_proto + 1, end, i_mouse},1);
        else
            ref_proto = nanmean(CC_total{i_proto + 1, i_proto, i_mouse},1);
        end
        CC_ref{i_proto, i_mouse} = ref_proto;
        
        % find normalization bounds for current prototype and animal
        % min: max of first baseline before any stim, max: max during any stim
        
        % extract stim and base indices for each run and prototype
        % (these vary for freeze and escape across runs...)
        
        tmp_protobase = ref_proto(idxBase);
        tmp_protobase = reshape(tmp_protobase',[],numtrial);
        tmp_protobase = nanmax(tmp_protobase,[],1);

        tmp_protostim = ref_proto(idxStim);
        tmp_protostim = reshape(tmp_protostim',[],numtrial);
        tmp_protostim = nanmax(tmp_protostim,[],1);
        
        minmax(i_proto, :, i_mouse) = [nanmax(tmp_protobase) nanmax(tmp_protostim)];
%         minmax(i_proto, :, i_mouse) = [max(ref_proto(idxBase)) max(ref_proto(idxStim))];
        
    end
end

figure(5); clf
for i_proto = 1:size(CC_ref,1)
    
    subplot(1,size(CC_ref,1),i_proto); hold on
    for i_mouse = 1:size(data.mouse,2)
        time = (1:size(CC_ref{i_proto,i_mouse},2))/20/60;
        if i_mouse == 1; plot([time(idxStim); time(idxStim)],[-.1 1.35],'color',[.7 1 1]); end 
        plot(time,CC_ref{i_proto,i_mouse} + (i_mouse-1)*.15)
    end
    
    set(gca,'ylim',[-.1 1.35], 'ytick', [0:size(data.mouse,2)-1]*0.15, 'yticklabel', strsplit(num2str(1:size(data.mouse,2))))
    ylabel('mouse #'); xlabel('time (min)')
    drawnow
end
subplot(1,3,1); title('disgust (quinine)')
subplot(1,3,2); title('pleasure (sucrose)')
subplot(1,3,3); title('pain (tail shock)')
%%
% perform normalization for each prototype within each animal
clear CC_total_norm
for i_mouse = 1:size(data.mouse,2)
    
    CC_total_mouse = CC_total(:,:,i_mouse);
    
    clear CC_total_mouse_norm
    for i_stim = 1:size(CC_total_mouse,2)
        for i_proto = 1:size(CC_total_mouse,1)-1
            
            for i_run = 1:size(CC_total_mouse{i_proto+1, i_stim},1)
                CC_total_mouse_norm{i_proto, i_stim}(i_run,:) = (CC_total_mouse{i_proto+1, i_stim}(i_run,:) - minmax(i_proto, 1, i_mouse))./ diff(minmax(i_proto, :, i_mouse));
            end

        end
    end
    
    CC_total_norm(:,:,i_mouse) = CC_total_mouse_norm;
end

%%
figure(10); clf
for i_mouse = 1:size(data.mouse,2)
    
    for i_proto = 1:size(CC_total_norm,1)
        
        for i_stim = 1:3
            
            subplot(size(CC_total_norm,1),3,i_stim + (i_proto - 1)*3); hold on
            plot(nanmean(CC_total_norm{i_proto,i_stim,i_mouse},1) + (i_mouse-1)*.1)
            set(gca,'xtick','','ylim',[-2 2])
            
        end
    end
end


%%
% extract proto corr values during stimulus delivery and extract time series
stim_val = cell(size(CC_total_mouse_norm,1),3); proto_plot = cell(size(CC_total_mouse_norm,1),3);
for i_mouse = 1%:size(data.mouse,2)
    
    CC_total_mouse_norm = CC_total_norm(:,:,i_mouse);
    
    for i_stim = 1:size(CC_total_mouse_norm,2)
        for i_proto = 1:size(CC_total_mouse_norm,1)
            
            proto_tmp = CC_total_mouse_norm{i_proto,i_stim};
            proto_tmp(:,idxBase) = 0; % set baseline to zero
            proto_tmp(proto_tmp < 0) = 0; % set negative values to zero
            proto_stim = proto_tmp(:,idxStim);
            
            stim_val{i_proto, i_stim} = [stim_val{i_proto, i_stim}; nanmean(proto_stim,2)]; % average value per session during stim
            
            % for plotting take extended trial
            proto_extend = proto_tmp(:,idxExtend);
            proto_extend = reshape(proto_extend', 1, size(proto_extend,2), size(proto_extend,1));
            proto_extend = reshape(proto_extend, size(proto_extend,2)/numtrial, numtrial, size(proto_extend,3));
            proto_extend = squeeze(nanmean(proto_extend,2));
            
%             figure; plot(proto_extend);
%             pause
            
            proto_plot{i_proto, i_stim} = [proto_plot{i_proto, i_stim} proto_extend];
        end
    end
end


figure(11); clf
EXP = 2;
for i_proto = 1:size(CC_total_mouse_norm,1)
    subplot(size(CC_total_mouse_norm,1),4,1 + 4*(i_proto-1));
    plot(nanmean(proto_plot{i_proto,1},2),'color',[.4 0 .4],'linewidth', 1.5);
%     plot(exp(EXP*nanmean(proto_plot{i_proto,1},2))-1,'color',[.4 0 .4],'linewidth', 1.5);
    set(gca,'ylim',[0 1],'xtick',''); title('quinine');
    
    subplot(size(CC_total_mouse_norm,1),4,2 + 4*(i_proto-1));
    plot(nanmean(proto_plot{i_proto,2},2),'color',[0 .4 0],'linewidth', 1.5);
%     plot(exp(EXP*nanmean(proto_plot{i_proto,2},2))-1,'color',[0 .4 0],'linewidth', 1.5);
    set(gca,'ylim',[0 1],'xtick',''); title('sucrose')
    
    subplot(size(CC_total_mouse_norm,1),4,3 + 4*(i_proto-1));
    plot(nanmean(proto_plot{i_proto,3},2),'color',[.4 0 0],'linewidth', 1.5);
%     plot(exp(EXP*nanmean(proto_plot{i_proto,3},2))-1,'color',[.4 0 0],'linewidth', 1.5);
    set(gca,'ylim',[0 1],'xtick',''); title('tail shock')
    
    subplot(size(CC_total_mouse_norm,1),4,4 + 4*(i_proto-1)); boxplot([stim_val{i_proto,1}' stim_val{i_proto,2}' stim_val{i_proto,3}'],...
        [ones(1,size(stim_val{i_proto,1},1)) 2*ones(1,size(stim_val{i_proto,2},1)) 3*ones(1,size(stim_val{i_proto,3},1))],'whisker', 1000);
    set(gca,'ylim',[0 1])
end

