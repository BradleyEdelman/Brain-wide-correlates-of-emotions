function fe_proto_normalize(data, param)
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman


%%
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
    
    % find sucrose, quinine, and tail shock runs for reference to build reference prototype correlations
    solicit_idx{1} = find(contains(fold,'sucrose'));
    solicit_idx{2} = find(contains(fold,'quinine'));
    solicit_idx{3} = find(contains(fold,'tail_shock'));
    
    % Load all prototypes for all stimuli
    % % Emotion along rows (neutral, disgust, pleasure, pain)
    % % Stimulus along column (sucrose, quinine, tail shock)
    CC_total = cell(4,3); % hard code for now
    for i_stim = 1:size(solicit_idx,2)
        for i_fold = 1:size(solicit_idx{i_stim},2)
        
            save_file = [fold_analysis{solicit_idx{i_stim}(i_fold)} 'Proto_corr.mat'];
            if exist(save_file,'file')
                while ~exist('TMP','var'); TMP = load(save_file); end; load(save_file); clear TMP
            
                for i_proto = 1:size(CC,1)
                    CC(idxBad) = nan;
                    CC = CC - nanmean(CC(:,1:3600),2);
                    CC_total{i_proto, i_stim}{end + 1} = movmean(CC(i_proto,:),10);
                end
            end

        end
    end
    
    % ensure all same size
    for i_stim = 1:size(CC_total,2)
        tmp = CC_total{1,i_stim};
        tmp_size = cellfun(@(v)size(v,2), tmp);
        tmp_size = min(tmp_size);
        
        for i_proto = 1:size(CC_total,1)
            for i_run = 1:size(CC_total{i_proto, i_stim},2)
                
                CC_total{i_proto, i_stim}{i_run} = CC_total{i_proto, i_stim}{i_run}(1:tmp_size);
            end
        end
    end
    
    % raw, pre-normalized prototype correlation time courses
    figure(5); clf
    AX = [0 1];
    subplot(4,3,1); plot(nanmean(cat(1,CC_total{1,1}{:}),1)); set(gca,'ylim', AX); ylabel('neutral'); title('sucrose')
    subplot(4,3,4); plot(nanmean(cat(1,CC_total{2,1}{:}),1)); set(gca,'ylim', AX); ylabel('disgust');
    subplot(4,3,7); plot(nanmean(cat(1,CC_total{3,1}{:}),1)); set(gca,'ylim', AX); ylabel('pleasure');
    subplot(4,3,10); plot(nanmean(cat(1,CC_total{4,1}{:}),1)); set(gca,'ylim', AX); ylabel('pain');
    
    subplot(4,3,2); plot(nanmean(cat(1,CC_total{1,2}{:}),1)); set(gca,'ylim', AX); title('quinine')
    subplot(4,3,5); plot(nanmean(cat(1,CC_total{2,2}{:}),1)); set(gca,'ylim', AX);
    subplot(4,3,8); plot(nanmean(cat(1,CC_total{3,2}{:}),1)); set(gca,'ylim', AX);
    subplot(4,3,11); plot(nanmean(cat(1,CC_total{4,2}{:}),1)); set(gca,'ylim', AX);
    
    subplot(4,3,3); plot(nanmean(cat(1,CC_total{1,3}{:}),1)); set(gca,'ylim', AX); title('tail shock')
    subplot(4,3,6); plot(nanmean(cat(1,CC_total{2,3}{:}),1)); set(gca,'ylim', AX);
    subplot(4,3,9); plot(nanmean(cat(1,CC_total{3,3}{:}),1)); set(gca,'ylim', AX);
    subplot(4,3,12); plot(nanmean(cat(1,CC_total{4,3}{:}),1)); set(gca,'ylim', AX);
    
    % normalize prototype correlations to evoked min/max prototype corr
    
    % find min and max for each emotion during solicited run (pleasure - sucrose)
    
    % look at prototype corr during stim
    Lbase = param.hog_disp.lbase;
    Lstim = param.hog_disp.lstim;
    numtrial = param.hog_disp.numtrial;

    % extract indices of stim for analysis and display
    [idxStim, idxBase, idxExtend] = fe_idxExtract(3600, 2440, numtrial, Lstim*5, Lbase);
    
    tmp = nanmean(cat(1,CC_total{2,2}{:}),1);
    MM(1,:) = [nanmax(tmp(idxBase)) nanmax(tmp(idxStim))]; % disgust
    tmp = nanmean(cat(1,CC_total{3,1}{:}),1);
    MM(2,:) = [nanmax(tmp(idxBase)) nanmax(tmp(idxStim))]; % pleasure
    tmp = nanmean(cat(1,CC_total{4,3}{:}),1);
    MM(3,:) = [nanmax(tmp(idxBase)) nanmax(tmp(idxStim))]; % pain
    
    for i_stim = 1:size(CC_total,2)
        for i_proto = 2:size(CC_total)
            for i_run = 1:size(CC_total{i_proto, i_stim},2)
                tmp = CC_total{i_proto, i_stim}{i_run};
                CC_total_norm{i_proto, i_stim}{i_run} = (tmp - MM(i_proto-1, 1))./(MM(i_proto-1, 2) - MM(i_proto-1, 1));
            end
        end
    end
    
    % average across runs to account for potentially lost runs for one stimulus on one day
    for i_stim = 1:size(CC_total_norm,2)
        for i_proto = 2:size(CC_total_norm,1)
            
            tmp = nanmean(cat(1,CC_total_norm{i_proto,i_stim}{:}),1);
%             tmp = CC_total_norm{i_proto, i_stim};
            tmp(tmp < 0) = 0;
            CC_total_norm_ave{i_proto,i_stim} = tmp;
            
        end
    end
    
    figure(6); clf
    AX = [0 1];
    subplot(3,3,1); plot(CC_total_norm_ave{2,1}); set(gca,'ylim', AX); ylabel('disgust'); title('sucrose')
    subplot(3,3,4); plot(CC_total_norm_ave{3,1}); set(gca,'ylim', AX); ylabel('pleasure');
    subplot(3,3,7); plot(CC_total_norm_ave{4,1}); set(gca,'ylim', AX); ylabel('pain');
    
    subplot(3,3,2); plot(CC_total_norm_ave{2,2}); set(gca,'ylim', AX); title('quinine')
    subplot(3,3,5); plot(CC_total_norm_ave{3,2}); set(gca,'ylim', AX);
    subplot(3,3,8); plot(CC_total_norm_ave{4,2}); set(gca,'ylim', AX);
    
    subplot(3,3,3); plot(CC_total_norm_ave{2,3}); set(gca,'ylim', AX); title('tail shock')
    subplot(3,3,6); plot(CC_total_norm_ave{3,3}); set(gca,'ylim', AX);
    subplot(3,3,9); plot(CC_total_norm_ave{4,3}); set(gca,'ylim', AX);
    
            

    for i_stim = 1:size(CC_total,2)
        for i_proto = 2:size(CC_total)
%             for i_run = 1:size(CC_total{i_proto, i_stim},2)
                
                proto_stim_norm{i_proto, i_stim} = CC_total_norm_ave{i_proto, i_stim}(idxStim);
                tmp = CC_total_norm_ave{i_proto, i_stim}(idxExtend);
                tmp = reshape(tmp, [], numtrial);
                proto_stim_norm_plot(i_proto, i_stim, :, i_mouse) = mean(tmp,2);
                
%             end
        end
    end
    
    for i_stim = 1:size(proto_stim_norm,2)
        for i_proto = 2:size(proto_stim_norm,1)
            stim_proto(i_proto, i_stim, i_mouse) = nanmean(cat(2,proto_stim_norm{i_proto, i_stim}));
            
        end
    end
    
    
    pause
    
end
%%
figure(7); clf 
subplot(3,1,1); boxplot([squeeze(stim_proto(2,1,:)) squeeze(stim_proto(2,2,:)) squeeze(stim_proto(2,3,:))]);
set(gca,'ylim',[0 1], 'xticklabel', {'sucrose', 'quinine', 'tail shock'}); title('disgust')
subplot(3,1,2); boxplot([squeeze(stim_proto(3,1,:)) squeeze(stim_proto(3,2,:)) squeeze(stim_proto(3,3,:))])
set(gca,'ylim', [0 1], 'xticklabel', {'sucrose', 'quinine', 'tail shock'}); title('pleasure')
subplot(3,1,3); boxplot([squeeze(stim_proto(4,1,:)) squeeze(stim_proto(4,2,:)) squeeze(stim_proto(4,3,:))]) 
set(gca,'ylim', [0 1], 'xticklabel',{'sucrose', 'quinine', 'tail shock'}); title('pain')


figure(8); clf
subplot(3,3,1); plot(nanmean(squeeze(proto_stim_norm_plot(2,1,:,:)),2)); set(gca,'ylim',[0 1]);
subplot(3,3,2); plot(nanmean(squeeze(proto_stim_norm_plot(2,2,:,:)),2)); set(gca,'ylim',[0 1]);
subplot(3,3,3); plot(nanmean(squeeze(proto_stim_norm_plot(2,3,:,:)),2)); set(gca,'ylim',[0 1]);

subplot(3,3,4); plot(nanmean(squeeze(proto_stim_norm_plot(3,1,:,:)),2)); set(gca,'ylim',[0 1]);
subplot(3,3,5); plot(nanmean(squeeze(proto_stim_norm_plot(3,2,:,:)),2)); set(gca,'ylim',[0 1]);
subplot(3,3,6); plot(nanmean(squeeze(proto_stim_norm_plot(3,3,:,:)),2)); set(gca,'ylim',[0 1]);

subplot(3,3,7); plot(nanmean(squeeze(proto_stim_norm_plot(4,1,:,:)),2)); set(gca,'ylim',[0 1]);
subplot(3,3,8); plot(nanmean(squeeze(proto_stim_norm_plot(4,2,:,:)),2)); set(gca,'ylim',[0 1]);
subplot(3,3,9); plot(nanmean(squeeze(proto_stim_norm_plot(4,3,:,:)),2)); set(gca,'ylim',[0 1]);

figure(9); clf
subplot(3,3,1); plot(squeeze(proto_stim_norm_plot(2,1,:,:))); set(gca,'ylim',[0 1]);
subplot(3,3,2); plot(squeeze(proto_stim_norm_plot(2,2,:,:))); set(gca,'ylim',[0 1]);
subplot(3,3,3); plot(squeeze(proto_stim_norm_plot(2,3,:,:))); set(gca,'ylim',[0 1]);

subplot(3,3,4); plot(squeeze(proto_stim_norm_plot(3,1,:,:))); set(gca,'ylim',[0 1]);
subplot(3,3,5); plot(squeeze(proto_stim_norm_plot(3,2,:,:))); set(gca,'ylim',[0 1]);
subplot(3,3,6); plot(squeeze(proto_stim_norm_plot(3,3,:,:))); set(gca,'ylim',[0 1]);

subplot(3,3,7); plot(squeeze(proto_stim_norm_plot(4,1,:,:))); set(gca,'ylim',[0 1]);
subplot(3,3,8); plot(squeeze(proto_stim_norm_plot(4,2,:,:))); set(gca,'ylim',[0 1]);
subplot(3,3,9); plot(squeeze(proto_stim_norm_plot(4,3,:,:))); set(gca,'ylim',[0 1]);


