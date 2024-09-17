function fus_behavior_bout_time_decoding(param)
%% Performing time-resolving region-based decoding of VBA behaviors

% Mace and Gogolla Labs - Max Planck Institute of Neurobiology
% Author: Bradley Edelman and Paulina Wanken
% Date: 11.04.22

%%
rewrite = param.decoding.rewrite;

% define save path/file for loading bout behavior and fus information
save_fold = param.save_fold;
save_dir = [save_fold 'behavior_fus_region\'];
if ~exist(save_dir,'dir'); mkdir(save_dir); end
results_file = [save_dir 'behavior_fus_region_' param.behavior.labels '_behav_labels.mat'];

if exist(results_file,'file')

    % load behavior label meanings
    label_meanings = param.behavior.label_meanings;
    
    load(results_file);
    
    % open behavior bout cells into a format easier to work with
    I_bout_total = cat(2,I_bout_total{:});
    I_bout_total_comp = cell(1, size(label_meanings,1));
    for i_behavior = 1:size(label_meanings,1)
        I_bout_total_comp{i_behavior} = cat(3,I_bout_total{i_behavior,:});
    end
    
    % find min number of trials (measurments) for a behavuor
    min_trial_cnt = min(cellfun(@(v) size(v,3), I_bout_total_comp));
    
    %%
    % perform one-vs-all decoding for all frames and regions
    idx = 1:size(I_bout_total_comp,2);
    for i_behavior = 1:size(I_bout_total_comp,2)
        fprintf('\n %s: ', label_meanings{i_behavior});
        
        % extract trials from behavior of interest
        idx_ONE = i_behavior;
        trials_ONE = I_bout_total_comp{idx_ONE};
        if size(trials_ONE, 3) > min_trial_cnt
            rand_idx = randperm(size(trials_ONE,3));
            rand_idx = rand_idx(1:min_trial_cnt);
            trials_ONE = trials_ONE(:,:,rand_idx);
        end
        
        % extract trials from other behaviors (evenly ish distributed)
        idx_ALL = idx; idx_ALL(i_behavior) = [];
        min_trial_cnt_ALL = ceil(min_trial_cnt/size(idx_ALL,2));
        
        trials_ALL = cell(0);
        for i_all = 1:size(idx_ALL,2)
            trials_ALL{i_all} = I_bout_total_comp{idx_ALL(i_all)};
            rand_idx = randperm(size(trials_ALL{i_all},3));
            rand_idx = rand_idx(1:min_trial_cnt_ALL);
            trials_ALL{i_all} = trials_ALL{i_all}(:,:,rand_idx);
        end
        trials_ALL = cat(3, trials_ALL{:});
        
        % time resolved decoding for each region (with balanced one and all behaviors)
        for i_reg = 1:size(trials_ALL,1)
            fprintf('%2.0f ', i_reg)
            
            for i_fr = 1:size(trials_ALL,2)
                
                current_ONE = squeeze(trials_ONE(i_reg, i_fr, :)); % extract data from current region and current frame
                current_ONE_rand = current_ONE(randperm(size(current_ONE,1))); % randomize order of class data
                
                current_ALL = squeeze(trials_ALL(i_reg, i_fr, :));
                current_ALL_rand = current_ALL(randperm(size(current_ALL,1)));
                
                fold_trial_cnt = floor(min_trial_cnt/5); % determine test set size for each corss-validation fold
                
                % five-fold cross validation
                for i_fold = 1:5
                    
                    % train model on 80% of data, test on 20%
                    train_idx = 1:min_trial_cnt;
                    test_idx = 1 + fold_trial_cnt*(i_fold-1):fold_trial_cnt + fold_trial_cnt*(i_fold-1); % take different test indices every fold
                    train_idx(test_idx) = []; % remove test indiced from train indices
                    
                    % organize train data for current fold
                    current_ONE_fold = current_ONE_rand(train_idx); % select train data for current fold
                    labels_ONE = zeros(size(current_ONE_fold,1),1); % create class labels
                    current_ALL_fold = current_ALL_rand(train_idx);
                    labels_ALL = ones(size(current_ALL_fold,1),1);
                    
                    % train model for current fold
                    LDA = fitcdiscr([current_ONE_fold; current_ALL_fold], [labels_ONE; labels_ALL], 'DiscrimType', 'quadratic');
                    
                    % organize test data for current fold
                    current_ONE_fold = current_ONE_rand(test_idx); % select train data for current fold
                    labels_ONE = zeros(size(current_ONE_fold,1),1); % create class labels
                    current_ALL_fold = current_ALL_rand(test_idx);
                    labels_ALL = ones(size(current_ALL_fold,1),1);
                    
                    % predict test set
                    out = predict(LDA,[current_ONE_fold;current_ALL_fold]);
                    
                    % compare predictions with test labels
                    compare = out - [labels_ONE; labels_ALL];
                    acc_fold(i_fold) = size(find(compare == 0),1)/size(compare,1);
                end
                
                acc_CV(i_reg, i_fr, i_behavior) = mean(acc_fold);
                [h,p] = ttest(acc_fold,.5,'alpha',.05);
                acc_p(i_reg, i_fr, i_behavior) = p;
            end
        end
    end
    
    %%
    % fdr correct time resolved p values on individual region basis...?
    for i_behavior = 1:size(I_bout_total_comp,2)
        for i_reg = 1:size(acc_p,1)
            acc_p_fdr(i_reg,:,i_behavior) = mafdr(acc_p(i_reg,:,i_behavior)');
        end
    end

%     for i_behavior = 1:size(I_bout_total_comp,2)
%         tmp = reshape(acc_p(:,:,i_behavior),[],1);
%         tmp_fdr = mafdr(tmp,'BHFDR',true);
%         acc_p_fdr2(:,:,i_behavior) = reshape(tmp_fdr, size(acc_p,1), size(acc_p,2));
%     end
    
    
    %%
    figure(30); clf; figure(31); clf; figure(32); clf
    Xvals = -param.behavior.bout_window:param.behavior.bout_window/2:param.behavior.bout_window;
    
    for i_behavior = 1:size(label_meanings,1)
        figure(30);
        subplot(2,2,i_behavior)
        imagesc(acc_CV(:,:,i_behavior));
        ylabel('brain region #'); xlabel('time (frames)');
        set(gca,'xticklabel',num2cell(Xvals),'xtick', Xvals - min(Xvals))
        title([label_meanings{i_behavior} ' Acc. (%)']);
        caxis([.5 .75])
        
        figure(31);
        subplot(2,2,i_behavior)
        imagesc(acc_p_fdr(:,:,i_behavior));
        ylabel('brain region #'); xlabel('time (frames)');
        set(gca,'xticklabel',num2cell(Xvals),'xtick', Xvals - min(Xvals))
        title([label_meanings{i_behavior} ' Acc. P-values ']);
        caxis([0 .05])
        
        figure(32);
        subplot(2,2,i_behavior)
        tmp = acc_CV(:,:,i_behavior);
        tmp(acc_p_fdr(:,:,i_behavior) > .05) = nan;
        imagesc(tmp);
        ylabel('brain region #'); xlabel('time (frames)');
        set(gca,'xticklabel',num2cell(Xvals),'xtick', Xvals - min(Xvals))
        title([label_meanings{i_behavior} ' Acc. (%) (tresh) ']);
        caxis([.5 .75])
    end
       
end
        
        
        

    
    

    
    



