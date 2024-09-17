function fe_HOG_corr(data, param)
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman


rewrite = param.hog_analyze.rewrite;
d_thresh = param.hog_analyze.d_thresh;
badframe_thresh = param.hog_analyze.badfram_thresh;

EXP = param.hog_disp.exp;
Lbase = param.hog_disp.lbase;
Lstim = param.hog_disp.lstim;
numtrial = param.hog_disp.numtrial;

% extract indices of stim for analysis and display
[idxStim, idxBase, idxExtend] = fe_idxExtract(3600, 2440, numtrial, Lstim, Lbase);

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
    
    % specify crop and spout coord file from preprocessing
    coord_file = [fold_analysis{1} 'crop_coord.mat'];
    if exist(coord_file,'file')
        load(coord_file)
    end
    
    spout_coord_file = [fold_analysis{1} 'spout_coord.mat'];
    if exist(spout_coord_file,'file')
        load(spout_coord_file)
    end
    
    % load prototypes
    proto_fold = [data.hog_fold  data.mouse{i_mouse} '\'];
    filename = dir(proto_fold);
    filename = {filename.name};
    filename = filename(cellfun(@(s)~isempty(regexp(s,'PROTO')),filename));
    clear proto proto_name
    for i_proto = 1:size(filename,2)
       tmp = load([proto_fold filename{i_proto}]);
       fields = fieldnames(tmp);
       proto(i_proto,:) = tmp.(fields{1});
       proto_name{i_proto} = filename{i_proto}(7:end-4);
    end
    
    for i_fold = 1:size(fold,2)
        
        save_file = [fold_analysis{i_fold} 'Proto_corr.mat'];
        hog_file = [fold_analysis{i_fold} 'hogs.mat'];
        if exist(hog_file,'file') && ~exist(save_file,'file') ||...
            exist(hog_file,'file') && rewrite == 1 
            
            % establish color code for different stimuli
            if contains(hog_file,'quinine')
                C = [136 14 79];
            elseif contains(hog_file,'sucrose')
                C =  [46 125 50];
            elseif contains(hog_file,'water')
                C = [0 0 255];
            elseif contains(hog_file,'tail_shock')
                C = [150 0 0];
            else
                C = [0 0 0];
            end
            
            % load hogs and remove spout indices
            hog = fe_hogLoadAndSpoutRemoval(hog_file, param, spoutInfo);
            
            % load DLC to determine if frames should be rejected
            d_thresh2 = d_thresh(1);
            if contains(hog_file,'tail_shock'); d_thresh2 = d_thresh(2); end
            DLC_file = [fold_analysis{i_fold} 'invalid_indices_2paw.mat'];
            if exist(DLC_file,'file')
                
                load(DLC_file)
                
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
            
            % calculate how many indices are considered bad
            idxBadProp = size(idxBad,1)/(size(idxBad,1) + size(idxGood,1))*100
            
            % decide whether run is acceptable/unacceptable 
            file_accept = [fold_analysis{i_fold} 'CLEAN_PAWS_ACCEPT.txt'];
            if exist(file_accept,'file'); delete(file_accept); end
            file_reject = [fold_analysis{i_fold} 'DIRTY_PAWS_REJECT.txt'];
            if exist(file_reject,'file'); delete(file_reject); end
            if idxBadProp < badframe_thresh
                fopen(file_accept,'wt');
            else
                fopen(file_reject,'wt');
            end
            fclose all
            
            % Better exclusion based on paw position within crop coordinates???
%             p1 = pos(:,11); p1(pos(:,12) < 0.7) = nan;
%             find(p1 > cropInfo.coords(1,2) & p1 <cropInfo.coords(2,2))
            
            % label bad frames in various idx variables
            idxStimGood = ~ismember(idxStim, idxBad);
            idxBaseGood = ~ismember(idxBase, idxBad);
%             idxExtendGood = ~ismember(idxExtend, idxBad);
            
            idxStimC = idxStim; idxStimC(idxStimGood == 0) = [];
            idxBaseC = idxBase; idxBaseC(idxBaseGood == 0) = [];
            if max(idxStimC) > size(hog,1) || max(idxBaseC) > size(hog,1)
            else
                
                cluster_hog = hog([idxStimC, idxBaseC],:);

                labels = [ones(1,size(idxStimC,2)) zeros(1,size(idxBaseC,2))];
                [fc, leafOrder] = fe_clustergram(cluster_hog, labels, C);

                savefig(fc, [fold_analysis{i_fold} 'clustergram.fig'])
                saveas(fc, [fold_analysis{i_fold} 'clustergram.png'])

                time = (1:size(hog,1))/20/60;
                CC = zeros(size(proto,1), size(hog,1));
                CC_clean = zeros(size(proto,1), size(hog,1));
%                 CC_clean2 = CC_clean;
                CC_clean_tmp = zeros(size(proto,1), size(hog,1));
%                 CC_Extend = zeros(size(proto,1), size(idxExtend,2));
                for i_proto = 1:size(proto,1)
                    
                    f = figure(13); clf
                    
                    % correlate hogs with each prototype
                    parfor (i_hog = 1:size(hog,1), 6)
                        i_hog
                        ctmp = corrcoef(hog(i_hog,:), proto(i_proto,:));
                        CC(i_proto,i_hog) = ctmp(1,2);
                    end
                    
                    % raw correlation
                    subplot(3,1,1); hold on;
                    plot([time(idxBase); time(idxBase)],[.65 1],'color',[.75 .75 .75])
                    plot([time(idxStim); time(idxStim)],[.65 1],'color','c')
                    plot(time,CC(i_proto,:),'k');
                    title('Original')
                    
                    % raw correlation with bad frames labels
                    subplot(3,1,2); hold on;
                    plot([time(idxBase); time(idxBase)],[.75 1],'color',[.75 .75 .75])
                    plot([time(idxStim); time(idxStim)],[.75 1],'color','c')
                    if ~isempty(idxBad)
                        plot([time(idxBad); time(idxBad)],[.75 1],'color','r')
                    end
                    plot(time,CC(i_proto,:),'k');
                    title('Bad frames labeled')
                    
                    % raw correlation with bad frames removed (set to nan)
                    subplot(3,1,3); hold on;
                    plot([time(idxBase); time(idxBase)],[.75 1],'color',[.75 .75 .75])
                    plot([time(idxStim); time(idxStim)],[.75 1],'color','c')
                    CC_clean(i_proto,:) = CC(i_proto,:); CC_clean(i_proto,idxBad) = nan;
                    plot(time,CC_clean(i_proto,:),'k');
                    title('Bad frames removed')
                    
%                     CC_clean_tmp(i_proto,:) = CC_clean(i_proto,:);
%                     
%                     CC_clean(i_proto,:) = (CC_clean(i_proto,:) - min(CC_clean(i_proto,:)))...
%                         /(max(CC_clean(i_proto,:)) - min(CC_clean(i_proto,:)));

%                     CC_Extend(i_proto,:) = CC_clean(i_proto,idxExtend);

%                     suptitle(proto_name{i_proto})
                    savefig(f, [fold_analysis{i_fold} 'Corr_with_' proto_name{i_proto} '.fig'])
                    saveas(f, [fold_analysis{i_fold} 'Corr_with_' proto_name{i_proto} '.png'])

                end

                fprintf('\nSaving: %s\n', save_file);
                save(save_file, 'CC', 'CC_clean', 'idxBad')
%                 save(save_file, 'CC', 'CC_clean', 'CC_Extend', 'idxBad')
                
                % Plot stimulus-locked prototypes
                
% %                 % isolate time around stimulus
% %                 CC_Extend = CC_clean_tmp(:,idxExtend);
% %                 % average across trials
% %                 CC_Extend = mean(reshape(CC_Extend,size(proto,1),[],numtrial),3);
% %                 % remove baseline (1 baseline length before stim)
% %                 CC_Extend = CC_Extend - repmat(nanmean(CC_Extend(:,Lbase+1:2*Lbase),2),1,size(CC_Extend,2));
% %                 % min-max normalize between lowest (0) and highest corr value
% %                 CC_Extend = (CC_Extend - 0)./(max(CC_Extend(:)) - 0);
% %                 
% %                 t1 = -2*Lbase:10*Lstim;
% %                 C = [150 150 150; 136 14 79; 46 125 50; 200 0 0]/255;
% %                 f1 = figure(15); clf; hold on
% %                 for i_proto = 1:size(proto,1)
% %                     plot(t1,movmean(CC_Extend(i_proto,:),10),...
% %                         'color', C(i_proto,:),'linewidth',1)
% %                 end
% %                 plot([t1(2*Lbase+1:2*Lbase+2*Lstim); t1(2*Lbase+1:2*Lbase+2*Lstim)],[.9 1],'color',[1 .5 0]);
% %                 set(gca,'ylim',[-1 1],'xlim',[t1(1) t1(end)]);
% %                 xlabel('time (sec)'); ylabel('proto. corr.');
% %                 
% %                 savefig(f1, [fold_analysis{i_fold} 'Proto_corr.fig'])
% %                 saveas(f1, [fold_analysis{i_fold} 'Proto_corr.png'])
            end
        end
    end
end
        

% 
% f1 = figure(15); clf
% t1 = -2*Lbase:10*Lstim; t2 = [t1, fliplr(t1)];
% for i_proto = 1:size(proto,1)
% 
%     subplot(2,2,i_proto);
%     if strcmp(proto_name{i_proto},'neutral')
%         C = [150 150 150];
%     elseif strcmp(proto_name{i_proto},'quinine')
%         C = [136 14 79];
%     elseif strcmp(proto_name{i_proto},'sucrose')
%         C =  [46 125 50];
%     elseif strcmp(proto_name{i_proto},'tail_shock')
%         C = [150 0 0];
%     end
% 
%     CC_plot = reshape(CC_Extend(i_proto,:),[],numtrial)';
%     M = nanmean(CC_plot,1);
%     S = nanstd(CC_plot,[],1);
%     between = [M + S, fliplr(M - S)];
%     fill(t2/20,between,C/255,'edgecolor','none','facealpha',0.5); hold on
%     plot(t1/20, M, 'color', C/255);
%     set(gca,'ylim',[0 1]);
%     xlabel('time (sec)'); ylabel(['Proto Similarity ^' num2str(EXP)]);
%     title(proto_name{i_proto})
%     drawnow
% end
%         