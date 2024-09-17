function fe_average_proto(data, param)
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman


% rewrite = param.ave_proto.rewrite;

EXP = param.hog_disp.exp;
Lbase = param.hog_disp.lbase;
Lstim = param.hog_disp.lstim;
numtrial = param.hog_disp.numtrial;

% extract indices of stim for analysis and display
[idxStim, idxBase, idxExtend] = fe_idxExtract(3600, 2440, numtrial, Lstim, Lbase);

CC_norm_grp = cell(3,1);
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
    
    
    for i_fold = 1:size(fold,2)
        
        save_file = [fold_analysis{i_fold} 'Proto_corr.mat'];
        
        if exist(save_file,'file')
            
            load(save_file)
            
            CC(:,idxBad) = nan;
            CC = CC(:,idxExtend);
            for i_proto = 1:size(CC,1)
                CC2(:,:,i_proto) = reshape(CC(i_proto,:),[],numtrial)';
            end
            
            CC2_M = squeeze(nanmean(CC2,1));
            CC2_S = squeeze(nanstd(CC2,[],1));
            
            figure(12);clf; subplot(1,2,1); plot(movmean(exp(EXP*CC2_M(:,2:end)),10))
            CC2_M = movmean(exp(EXP*CC2_M),10);
            
            % subtract baseline
            base = CC2_M(1:Lbase*2,:);
            CC2_M = CC2_M - repmat(nanmean(base,1),size(CC2_M,1),1);
            
            base2 = CC2_M(1:Lbase,2:end);
            absMin = mean(base2(:));
            trial = CC2_M(Lbase*2+1:end,2:end);
            absMax = max(trial(:));
            
            for i_proto = 1:size(CC,1)
                CC_norm(i_proto,:) = (CC2_M(:,i_proto) - absMin)./(absMax - absMin);
            end
            subplot(1,2,2); plot(CC_norm(2:end,:)')
            
            if contains(save_file,'quinine')
                IDX = 1; title('quinine')
            elseif contains(save_file,'sucrose')
                IDX = 2; title('sucrose')
            elseif contains(save_file,'tail_shock')
                IDX = 3; title('tail_shock')
            end

            CC_norm_grp{IDX}(:,:,end+1) = CC_norm;
            pause(2)
            
        end
    end
end
         

C = [150 150 150; 136 14 79; 46 125 50; 150 0 0];
t1 = -2*Lbase:10*Lstim;
t2 = [t1, fliplr(t1)];
figure(14); clf; figure(15); clf
for i_proto = 1:3
    for j_proto = 1:4
        
        data_tmp = squeeze(CC_norm_grp{i_proto}(j_proto,:,2:end));
        bad = max(abs(data_tmp),[],1);
        data_tmp2 = data_tmp; data_tmp2(:,bad > 4) = [];
        M = nanmean(data_tmp2,2)';
        S = (nanstd(data_tmp2,[],2)')./sqrt(size(data_tmp2,2)-1);
        
        between = [M + S, fliplr(M - S)];
        
        figure(14)
        subplot(4,3,3*(j_proto-1) + i_proto); hold on
        plot([t1(2*Lbase+1:2*Lbase+2*Lstim); t1(2*Lbase+1:2*Lbase+2*Lstim)],[0.75 1],'color','c')
        fill(t2,between,C(j_proto,:)/255,'edgecolor','none','facealpha',0.5);
        plot(t1, M, 'color', C(j_proto,:)/255,'linewidth',1);
        set(gca,'ylim',[-1 1],'xticklabel',''); grid on
        
        figure(15)
        subplot(4,3,3*(j_proto-1) + i_proto); hold on
        plot([t1(2*Lbase+1:2*Lbase+2*Lstim); t1(2*Lbase+1:2*Lbase+2*Lstim)],[2.75 3],'color','c')
        plot(t1, data_tmp,'color',C(j_proto,:)/255)
        set(gca,'ylim',[-4 4],'xticklabel','')
        
    end
end

for ii = 1:2
    figure(13+ii)
    subplot(4,3,1); title('Bitter')
    subplot(4,3,2); title('Sweet')
    subplot(4,3,3); title('Tailshock')
    subplot(4,3,1); ylabel('Neutral')
    subplot(4,3,4); ylabel('Disgust')
    subplot(4,3,7); ylabel('Pleasure')
    subplot(4,3,10); ylabel('Pain')
end
            
            
            
        
        
        