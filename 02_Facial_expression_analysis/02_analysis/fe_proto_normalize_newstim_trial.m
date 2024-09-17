function fe_proto_normalize_newstim_trial(data, param)
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman


%%
proto_stim = param.stim.proto;
new_stim = param.stim.new;

for i_mouse = 1:size(data.mouse,2)
    i_mouse
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
    
    % find protoype runs for reference to build reference prototype correlations
    for i_protostim = 1:size(proto_stim,2)
        proto_idx{i_protostim,1} = proto_stim{i_protostim};
        proto_idx{i_protostim,2} = data.idxProto(i_mouse,i_protostim);
    end
    
    % find new stim runs to compare to to reference prototypes
    for i_newstim = 1:size(new_stim,2)
        new_idx{i_newstim,1} = new_stim{i_newstim};
        new_idx{i_newstim,2} = find(contains(fold, new_stim{i_newstim}));
        if size(new_idx{i_newstim,2},2) > 1
            new_idx{i_newstim,2} = new_idx{i_newstim,2}(end);
        end
    end
    
    % Load reference prototype correlation for each stimulus
    for i_protostim = 1:size(proto_stim,2)
        
        proto_file = [fold_analysis{proto_idx{i_protostim,2}} 'Proto_corr.mat'];
        accept_file = [fold_analysis{proto_idx{i_protostim,2}} 'CLEAN_PAWS_ACCEPT.txt'];
        if exist(proto_file,'file')% && exist(accept_file,'file')
                
            while ~exist('TMP','var'); try TMP = load(proto_file); end; end
            load(proto_file); clear TMP
                
%             CC(:,idxBad) = nan; % remove bad frames
            CC = CC - nanmean(CC(:,1:3600),2); % remove mean of baseline
            % prototypes always saved as neutral, disgust, pleasure, pain in CC structure
            sm_factor = 10;
            switch proto_idx{i_protostim,1}
                case 'neutral'
                    CC_proto{i_protostim, i_mouse} = movmean(CC(1,:),sm_factor);
                case 'disgust'
                    CC_proto{i_protostim, i_mouse} = movmean(CC(2,:),sm_factor);
                case 'pleasure'
                    CC_proto{i_protostim, i_mouse} = movmean(CC(3,:),sm_factor);
                case 'pain'
                    CC_proto{i_protostim, i_mouse} = movmean(CC(4,:),sm_factor);
            end
        end
    end
    
    % Load prototype correlations for all new stim
    for i_newstim = 1:size(new_stim,2)
        
        proto_file = [fold_analysis{new_idx{i_newstim,2}} 'Proto_corr.mat'];
        accept_file = [fold_analysis{new_idx{i_newstim,2}} 'CLEAN_PAWS_ACCEPT.txt'];
        if exist(proto_file,'file')%% && exist(accept_file,'file')
                
            while ~exist('TMP','var'); try TMP = load(proto_file); end; end
            load(proto_file); clear TMP
                
%             CC(:,idxBad) = nan; % remove bad frames
            CC = CC - nanmean(CC(:,1:3600),2); % remove mean of baseline
            
            % prototypes always saved as neutral, disgust, pleasure, pain in CC structure
            % % extract all prototype correlations for each new stim
            for i_protostim = 1:size(proto_stim,2)
                switch proto_idx{i_protostim,1}
                    case 'neutral'
                        CC_new{i_protostim, i_newstim, i_mouse} = movmean(CC(1,:),sm_factor);
                    case 'disgust'
                        CC_new{i_protostim, i_newstim, i_mouse} = movmean(CC(2,:),sm_factor);
                    case 'pleasure'
                        CC_new{i_protostim, i_newstim, i_mouse} = movmean(CC(3,:),sm_factor);
                    case 'pain'
                        CC_new{i_protostim, i_newstim, i_mouse} = movmean(CC(4,:),sm_factor);
                end
            end
            
        end
    end
    
end

% ensure that prototype time course is the same for each stimulus within each mouse
% % (some recordings drop a few frames so cant stack time courses so easily)

[b, a] = butter(5,0.01/20,'high');
for i_mouse = 1:size(CC_proto,2)
    
    % do this for the protoype time series
    CC_proto_mouse = CC_proto(:,i_mouse);
    sz = cellfun(@(v)size(v,2), CC_proto_mouse);
    sz = min(sz);
    CC_proto_mouse = cellfun(@(v)v(1:sz), CC_proto_mouse, 'uniformoutput', false);
    CC_proto_mouse = cellfun(@(v)fillmissing(v,'linear'), CC_proto_mouse, 'uniformoutput', false);
    CC_proto_mouse = cellfun(@(v)filtfilt(b,a,v), CC_proto_mouse, 'uniformoutput', false);
    CC_proto(:,i_mouse) = CC_proto_mouse;
    
    CC_new_mouse = CC_new(:,:,i_mouse);
    CC_new_mouse = cellfun(@(v)fillmissing(v,'linear'), CC_new_mouse, 'uniformoutput', false);
    CC_new_mouse = cellfun(@(v)filtfilt(b,a,v), CC_new_mouse , 'uniformoutput', false);
    CC_new(:,:,i_mouse) = CC_new_mouse;
    
end

% extract min max values for each mouse and each prototype
Lbase = param.hog_disp.lbase;
Lstim = param.hog_disp.lstim;
numtrial = param.hog_disp.numtrial;

% extract indices of stim and baseline
[idxStim, idxBase, idxExtend] = fe_idxExtract(3600, 2440, 5, Lstim, Lbase);

for i_mouse = 1:size(CC_proto,2)
    
    CC_proto_mouse = cat(1,CC_proto{:,i_mouse});
    for i_proto = 1:size(CC_proto_mouse,1)
        tmp_protobase = CC_proto_mouse(i_proto,idxBase);
        tmp_protobase = reshape(tmp_protobase',[],numtrial);
        tmp_protobase = nanmean(tmp_protobase,1);
        tmp_protostim = CC_proto_mouse(i_proto,idxStim);
        tmp_protostim = reshape(tmp_protostim',[],numtrial);
        tmp_protostim = nanmean(tmp_protostim,1);
        
        MM(i_proto, :, i_mouse) = [nanmax(tmp_protobase) nanmax(tmp_protostim)];
    end
    
end

% perform normalization on new stimuli
for i_mouse = 1:size(CC_new,3)
    
    CC_new_tmp = CC_new(:,:,i_mouse);
    for i_stim = 1:size(CC_new_tmp,2)
        for i_proto = 1:size(CC_new_tmp,1)
            
            CC_new_tmp{i_proto, i_stim} = (CC_new_tmp{i_proto, i_stim} - MM(i_proto,1,i_mouse))./...
                diff(MM(i_proto,:,i_mouse));
            
        end
    end
    CC_new(:,:,i_mouse) = CC_new_tmp;
end

for i_mouse = 1:size(CC_new,3)
    
    CC_new_tmp = CC_new(:,:,i_mouse);
    
    figure(10 + i_mouse); clf
    for i_stim = 1:size(CC_new,2)
        if ~isempty(CC_new_tmp{1,i_stim})
            
            F = [10 + i_mouse, size(CC_new_tmp,2), size(CC_proto,1) + 1, 1 + (size(CC_proto,1) + 1)*(i_stim - 1)];
            tmp = CC_new_tmp{1,i_stim};
            tmp(tmp < 0) = 0; tmp(idxBase) = 0;
            tmp = tmp(idxExtend); tmp = reshape(tmp,[],numtrial); 
            new_trace{1,i_stim,i_mouse} = tmp; tmp = nanmean(tmp,2);
            plot_areaerrorbar(tmp',[], F); ylim([0 1])

            F = [10 + i_mouse, size(CC_new_tmp,2), size(CC_proto,1) + 1, 2 + (size(CC_proto,1) + 1)*(i_stim - 1)];
            tmp = CC_new_tmp{2,i_stim};
            tmp(tmp < 0) = 0; tmp(idxBase) = 0;
            tmp = tmp(idxExtend); tmp = reshape(tmp,[],numtrial);
            new_trace{2,i_stim,i_mouse} = tmp; tmp = nanmean(tmp,2);
            plot_areaerrorbar(tmp', [], F); ylim([0 1])

            subplot(size(CC_new_tmp,2), size(CC_proto,1) + 1, 3 + (size(CC_proto,1) + 1)*(i_stim - 1));
            tmp = cat(1,CC_new_tmp{:,i_stim});
            tmp(tmp < 0) = 0; tmp = tmp(:,idxStim);
            tmp = reshape(tmp',[],numtrial,2); tmp = squeeze(nanmean(tmp,1));
            new_trace{3,i_stim,i_mouse} = tmp;
            boxplot(tmp); ylim([0 1])
        end
    end
end
    
    
    


    
    

