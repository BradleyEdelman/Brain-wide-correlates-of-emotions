function fe_proto_normalize_3(data, param)
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman


%% LOAD REFERENCE PROTOTYPE FOR EACH STIMULUS
order = {'neutral', 'quinine', 'sucrose', 'tail_shock', 'freeze', 'escape'};
proto_color = [158 159 162; 144 36 94; 60 152 78; 228 88 55; 33 56 105; 65 111 185];
% order = {'neutral', 'quinine', 'sucrose', '2PE', 'peanut_oil', 'isoP'};
% proto_color = [158 159 162; 144 36 94; 60 152 78; 255 155 50; 255 155 50; 150 100 50];

CC_total = cell(size(order,2),size(order,2)-1,size(data.mouse,2));
idxStim = cell(size(order,2),size(order,2)-1,size(data.mouse,2));
idxBase = cell(size(order,2),size(order,2)-1,size(data.mouse,2));
idxExtend = cell(size(order,2),size(order,2)-1,size(data.mouse,2));
for i_mouse = 1:size(data.mouse,2)
    i_mouse
    
    % specify image folders and corresponding hog folders to save to
    [fold_face, fold_face_hog] = fe_identify_video_fold(data, i_mouse, 'FACE');
    
    % find STIMULUS (not prototype) runs for reference
    solicit_idx = cell(0);
    for i_stim = 2:size(order,2) % skip neutral
        stim_idx = find(contains(fold_face,order{i_stim}));
        solicit_idx{i_stim - 1} = stim_idx;
    end
    
    % Load only reference prototype for each stimulus (i.e. pleasure for sucrose)
    for i_stim = 1:size(solicit_idx,2)
        for i_fold = 1:size(solicit_idx{i_stim},2)
        
            proto_file = [fold_face_hog{solicit_idx{i_stim}(i_fold)} 'Proto_corr_all.mat'];
            accept_file = [fold_face_hog{solicit_idx{i_stim}(i_fold)} 'CLEAN_PAWS_ACCEPT.txt'];
            if exist(proto_file,'file') && exist(accept_file,'file')
                
                [CC, stim, base, extend] = fe_arrange_prototype_info(proto_file, order);
                if ~isempty(CC)
                    
                    if param.hog_analyze.exp == 1
                        EXP = param.hog_analyze.exp_val;
                        CC = exp(EXP*CC);
                    end
                    
                    CC = CC - nanmean(CC(:,1:3600),2); % remove mean of baseline

                    % prototypes should be neutral, disgust, pleasure, pain, freeze, escape
                    for i_proto = 1:size(CC,1)
                        tmp = CC(i_proto,:);
                        CC_total{i_proto, i_stim, i_mouse}{end+1} = tmp;
                        idxStim{i_proto, i_stim, i_mouse}{end+1} = stim(i_proto);
                        idxBase{i_proto, i_stim, i_mouse}{end+1} = base(i_proto);
                        idxExtend{i_proto, i_stim, i_mouse}{end+1} = extend(i_proto);
                    end
                    
                end
            else
                accept_file
            end
        end
    end
end

%% Ensure that prototype time course is the same lenght for each stimulus within each mouse
% (some recordings drop a few frames so cant stack time courses so easily)
[b, a] = butter(5,0.01/20,'high');
clear CC_tmp_uniform_length CC_total_uniform
for i_mouse = 1:size(data.mouse,2)
    
    CC_total_mouse = CC_total(:,:,i_mouse);
    empty_row = cellfun('isempty',CC_total_mouse(:,1));
    CC_total_mouse(empty_row,:) = [];
    for i_stim = 1:size(CC_total_mouse,2)
        for i_proto = 1:size(CC_total_mouse,1)
            
            CC_tmp = cat(2, CC_total_mouse{i_proto, i_stim});
            if ~isempty(CC_tmp)
                sz = cellfun(@(v)size(v,2), CC_tmp);
                sz = min(sz);
                CC_tmp = cellfun(@(v)v(1:sz), CC_tmp, 'uniformoutput', false);
                CC_tmp = cellfun(@(v)fillmissing(v,'linear'), CC_tmp, 'uniformoutput', false);
                CC_tmp = cellfun(@(v)filtfilt(b,a,v), CC_tmp, 'uniformoutput', false);
                CC_tmp = cat(1, CC_tmp{:});
            end
            CC_tmp_uniform_length{i_proto, i_stim} = CC_tmp;
        end
    end
    
    if exist('CC_tmp_uniform_length','var')
        CC_total_uniform(:,:,i_mouse) = CC_tmp_uniform_length;
        clear CC_tmp_uniform_length
    end
end
CC_total = CC_total_uniform;

% detect empty columns from all mice and remove
for i_data = size(CC_total,2):-1:1

    d = squeeze(cat(2,CC_total(:,i_data,:)));d = d(:);
    dcnt = size(find(cellfun(@isempty,d)==1),1);
    if dcnt == size(d,1)
        CC_total(:,i_data,:) = [];
    end
end

%% Plot "solicited" prototype time series/trial for each animal
CC_ref = cell(size(CC_total,1), size(CC_total,3));
for i_mouse = 1:size(CC_total,3)
    
    % first create reference prototype for each animal and emotion
    % reference prototypes are along the diagonal (after the first row - neutral)
    for i_proto = 1:size(CC_total,1)
        if i_proto == 1 % neutral (built on quinine runs)
            CC_ref{i_proto, i_mouse} = nanmean(CC_total{i_proto, 1, i_mouse},1);
        elseif i_proto > size(CC_total,2) % escape and freeze in the tail shock runs...
            CC_ref{i_proto, i_mouse} = nanmean(CC_total{i_proto, end, i_mouse},1);
        else
            CC_ref{i_proto, i_mouse} = nanmean(CC_total{i_proto, i_proto - 1, i_mouse},1);
        end
    end
end


figure(5); clf
numtrial = param.hog_disp.numtrial;
Lbase = param.hog_disp.lbase;
Lstim = param.hog_disp.lstim;
[stim, base, extend] = fe_idxExtract(3600, 2440, numtrial, Lstim, Lbase);
for i_proto = 1:size(CC_ref,1)
    
    subplot(1,size(CC_ref,1),i_proto); hold on
    for i_mouse = 1:size(CC_ref,2)
        time = (1:size(CC_ref{i_proto,i_mouse},2))/20/60;
        if i_mouse == 1; plot([time(stim); time(stim)],[-.1 1.35],'color',[.7 1 1]); end 
        plot(time,CC_ref{i_proto,i_mouse} + (i_mouse-1)*.15)
    end
    
    set(gca,'ylim',[-.1 1.35], 'ytick', [0:size(data.mouse,2)-1]*0.15, 'yticklabel', strsplit(num2str(1:size(data.mouse,2))))
    ylabel('mouse #'); xlabel('time (min)')
    title(order{i_proto});
    drawnow
end


%% Extract similarity scores (and min/max) for prototypes (trial and time series) during target condition trials

clear CC_pre_norm_TS CC_pre_norm_base CC_pre_norm_trial minmax minmax_trial
for i_mouse = 1:size(CC_total,3)
    for i_stim = 1:size(CC_total,1) % for each stimulus/condition

        if i_stim == 1 % baseline comes from quinine runs (1st column)
            Data_idx = 1;
        elseif i_stim > size(CC_total(:,:,i_mouse),2) % freeze and escape from tail shock runs (3rd colum)
            Data_idx = size(CC_total(:,:,i_mouse),2);
        else % "core" stim from 1st-3rd columns respectively
            Data_idx = i_stim - 1;
        end

        % all prototypes correlations from condition-specific data
        CC_condition = CC_total(:, Data_idx, i_mouse);
        EMPT = sum(cellfun(@isempty,CC_condition));
        if EMPT == 0 || isempty(EMPT)
            % baseline and stimulation indices for current condition trials
            stim_tmp = cat(2, idxStim{i_stim, Data_idx, i_mouse}{:});
            base_tmp = cat(2, idxBase{i_stim, Data_idx, i_mouse}{:});   
            stim_add = 3;
            % add  time onto the end of the stimulus indices to visualize longer trials
            % (remove trials that now extend longer than the length of the time series)
            for i_run = 1:size(stim_tmp,2)
                stim_tmp1 = reshape(stim_tmp{i_run}, Lstim, size(stim_tmp{i_run},2)/Lstim);
                base_tmp1 = reshape(base_tmp{i_run}, Lbase, size(base_tmp{i_run},2)/Lbase);
                if ~isempty(stim_tmp1)
                    clear stim_tmp2
                    for i_trial = 1:size(stim_tmp1,2)
                        stim_tmp2(:,i_trial) = stim_tmp1(end,i_trial) + 1:stim_tmp1(end,i_trial) + stim_add*(Lstim/2);
                    end
                    stim_tmp1 = [stim_tmp1; stim_tmp2];
                    long_trial = find(sum(stim_tmp1 > size(CC_condition{1},2),1) ~= 0);
                    stim_tmp1(:, long_trial) = []; base_tmp1(:, long_trial) = [];
                    stim_tmp{i_run} = stim_tmp1(:)'; base_tmp{i_run} = base_tmp1(:)';
                end
            end

            for i_proto = 1:size(CC_total(:,:,i_mouse),1) % for each prototype

                CC_tmp = CC_condition{i_proto}; % prototype similarity scores for current prototype
                CC_tmp_trial = [];
                % extract prototype scores for current condition trials
                for i_run = 1:size(CC_tmp,1)

                    CC_tmp_base = CC_tmp(i_run,base_tmp{i_run});
                    CC_tmp_base = reshape(CC_tmp_base', Lbase, size(CC_tmp_base,2)/Lbase);

                    CC_tmp_stim = CC_tmp(i_run,stim_tmp{i_run});
                    CC_tmp_stim = reshape(CC_tmp_stim', ((stim_add/2)+1)*Lstim, size(CC_tmp_stim,2)/(((stim_add/2)+1)*Lstim));

                    % add full trial indices (baseline, stimulus) from successive runs
                    CC_tmp_trial = [CC_tmp_trial [CC_tmp_base; CC_tmp_stim]];
                end

                % ensure that all baselines set to 0
                for i_trial = 1:size(CC_tmp_trial,2)
                    CC_tmp_trial(:,i_trial) = CC_tmp_trial(:,i_trial) - mean(CC_tmp_trial(1:Lbase,i_trial));
                end
        
                CC_tmp_trial(CC_tmp_trial < 0) = 0;

                CC_pre_norm_TS{i_proto, i_stim, i_mouse} = CC_tmp_trial;
                CC_pre_norm_base{i_proto, i_stim, i_mouse} = nansum(CC_tmp_trial(1:Lbase,:),1);
                CC_pre_norm_trial{i_proto, i_stim, i_mouse} = nansum(CC_tmp_trial(Lbase+1:Lbase+Lstim,:),1);
            end
        else
            CC_pre_norm_TS{i_proto, i_stim, i_mouse} = [];
            CC_pre_norm_base{i_proto, i_stim, i_mouse} = [];
            CC_pre_norm_trial{i_proto, i_stim, i_mouse} = [];
        end
    end
    
    % plot mouse data and extract min and max normalization values
    figure(46); clf
    for i_proto = 1:size(CC_pre_norm_TS,1)
        for i_stim = 1:size(CC_pre_norm_TS,2)
            subplot(size(CC_pre_norm_TS,1), size(CC_pre_norm_TS,2), i_stim + (i_proto-1)*size(CC_pre_norm_TS,2))
            
            CC_pre_norm_tmp_TS = movmean(CC_pre_norm_TS{i_proto,i_stim,i_mouse},10,1);
            CC_pre_norm_tmp_base = CC_pre_norm_base{i_proto,i_stim,i_mouse};
            CC_pre_norm_tmp_trial = CC_pre_norm_trial{i_proto,i_stim,i_mouse};
            if ~isempty(CC_pre_norm_tmp_TS)
                plot(CC_pre_norm_tmp_TS)
                if i_proto == i_stim
                    CC_pre_norm_tmp_TS = nanmax(CC_pre_norm_tmp_TS,[],2);
                    minmax(i_proto,1,i_mouse) = nanmax(CC_pre_norm_tmp_TS(1:Lbase,:));
                    minmax(i_proto,2,i_mouse) = nanmax(CC_pre_norm_tmp_TS(Lbase+1:Lbase+Lstim,:));

                    minmax_trial(i_proto,1,i_mouse) = nanmax(CC_pre_norm_tmp_base);
                    minmax_trial(i_proto,2,i_mouse) = nanmax(CC_pre_norm_tmp_trial);
                end
            else
                if i_proto == i_stim
                    minmax(i_proto,1:2,i_mouse) = nan;
                    minmax_trial(i_proto,1:2,i_mouse) = nan;
                end
            end
        end
    end

    minmax(:,:,i_mouse) = sort(minmax(:,:,i_mouse),2,'ascend');
    minmax_trial(:,:,i_mouse) = sort(minmax_trial(:,:,i_mouse),2,'ascend');
    for i_proto = 1:size(CC_pre_norm_TS,1)
        for i_stim = 1:size(CC_pre_norm_TS,2)
            subplot(size(CC_pre_norm_TS,1), size(CC_pre_norm_TS,2), i_stim + (i_proto-1)*size(CC_pre_norm_TS,2))
            set(gca,'ylim',[-0.1 max(minmax(:,2,i_mouse))],'xlim',[0 size(CC_pre_norm_TS{end},1)]);
        end
    end
    
    pause(1)
end


%%      
clear CC_post_norm_TS CC_post_norm_trial
for i_mouse = 1:size(CC_pre_norm_TS,3)

    CC_tmp_TS = CC_pre_norm_TS(:,:,i_mouse);    
    CC_tmp_trial = CC_pre_norm_trial(:,:,i_mouse);
    for i_stim = 1:size(CC_tmp_TS,2)
        for i_proto = 1:size(CC_tmp_TS,1)
            
            % minmax(i_proto,1,i_mouse) = normalize within prototype
            % minax(i_stim,1,i_mouse) = normalie within stimulus
            CC_tmp_TS{i_proto,i_stim} = rescale(CC_tmp_TS{i_proto,i_stim},...
                'inputmin', minmax(i_proto,1,i_mouse), 'inputmax', minmax(i_proto,2,i_mouse));

            CC_tmp_trial{i_proto,i_stim} = rescale(CC_tmp_trial{i_proto,i_stim},...
                'inputmin', minmax_trial(i_proto,1,i_mouse), 'inputmax', minmax_trial(i_proto,2,i_mouse));
        end
    end

    CC_post_norm_TS(:,:,i_mouse) = CC_tmp_TS;
    CC_post_norm_trial(:,:,i_mouse) = CC_tmp_trial;

    figure(83); clf
    for i=1:size(CC_post_norm_TS,1)
        subplot(1, size(CC_post_norm_TS,1), i)
        plot(mean(CC_post_norm_TS{i,i,i_mouse},2));
        drawnow
    end
    
%     pause
end


%%
proto_name = {'neutral', 'disgust', 'pleasure', 'pain', 'pass. fear', 'act. fear'};
% proto_name = {'neutral', 'taste disg.', 'taste pleas.', 'odor pleas. 1', 'odor pleas. 2', 'odor disg.'}

IDX = 1:8;

p_thresh = 0.0000001;
YLIM = [0 0.35];

figure(63); clf
for i_proto = 1:size(CC_post_norm_TS,1)
    for i_stim = 1:size(CC_post_norm_TS,2)
        subplot(size(CC_post_norm_TS,1), size(CC_post_norm_TS,2), i_stim + (i_proto-1)*size(CC_post_norm_TS,2))
        
        tmp = squeeze(CC_post_norm_TS(i_proto,i_stim,IDX));
        
        tmp_trials = cat(2,tmp{:});
        tmp_trials = mean(tmp_trials(Lbase+1:Lbase+Lstim,:),1);
        norm_trials{i_proto,i_stim} = tmp_trials;

%         tmp = cellfun(@(v) nanmean(v,2), tmp, 'uniformoutput', false);
        tmp = cat(2, tmp{:});
        hold on
        plot([Lbase Lbase+Lstim; Lbase Lbase+Lstim],YLIM,'--','color',proto_color(i_stim,:)/255,'linewidth',1)
        

        M = movmean(nanmean(tmp,2),10)';
        S = movmean(nanstd(tmp,[],2)./sqrt(size(tmp,2)),10)';
        t1 = 1:size(M,2); t2 = [t1, fliplr(t1)];
        between = [M + S, fliplr(M - S)];
        fill(t2,between,proto_color(i_proto,:)/255,'edgecolor','none','facealpha',0.5); hold on
        plot(t1, M, 'color', proto_color(i_proto,:)/255,'linewidth',1.5);

        set(gca,'ylim',YLIM,'xlim',[0 200])
        if i_stim ~= 1; set(gca,'xtick','','ytick',''); else; ylabel(proto_name{i_proto}); end
        if i_proto == size(CC_post_norm_TS,1); set(gca,'xtick',[1 Lbase size(tmp,1)],...
            'xticklabel',round(([1 Lbase size(tmp,1)]-Lbase)/20)); xlabel('time (sec)'); end
        if i_proto == 1; title([order{i_stim} ' (n = ' num2str(size(norm_trials{1,i_stim},2)) ')']); end
    
    end
end
set(gcf,'color','w')


figure(64); clf
for i_proto = 2:size(CC_post_norm_TS,1)
    subplot(size(CC_post_norm_TS,1)-1,1,i_proto-1)
    
    val_box = []; label_box = []; %val_violin = cell(1,size(CC_post_norm_TS,2));
    for i_stim = 1:size(CC_post_norm_TS,2)
        
        val_box = [val_box cat(2,CC_post_norm_trial{i_proto,i_stim,IDX})];
        label_box = [label_box i_stim*ones(1,size(cat(2,CC_post_norm_trial{i_proto,i_stim,IDX}),2))];
        
%         val_violin{i_stim} = cat(2,CC_post_norm_trial{i_proto,i_stim,IDX});
        
        p(i_stim) = ranksum(cat(2,CC_post_norm_trial{i_proto,1,IDX}),cat(2,CC_post_norm_trial{i_proto,i_stim,IDX}));
    end
    pp = p; pp(p>p_thresh) = -1; pp(p<p_thresh) = 1.5;
    
    hold on

%     violin(val_violin,'facecolor', proto_color(i_proto,:)/255, 'facealpha',...
%         0.75,'edgecolor','k','bw',.05,'plotlegend', 0)

    boxplot(val_box,label_box,'symbol','.','whisker',1e10,'colors',repmat(proto_color(i_proto,:)/255,i_stim,1))
    scatter(1:size(pp,2),pp,15,'*','k')
    set(gca,'ylim',[-.25 2],'xtick','')
    ylabel(proto_name{i_proto})
end
set(gca,'xtick',1:size(CC_post_norm_TS,2),'xticklabel',order)
set(gcf,'color','w')
xtickangle(45)





