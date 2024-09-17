function fus_behavior_bout_time_decoding_MVPA_voxel(data_files,param)
%% Performing time-resolving region-based decoding of VBA behaviors

% Mace and Gogolla Labs - Max Planck Institute of Neurobiology
% Author: Bradley Edelman
% Date: 08.08.22

%%
rewrite_mvpa = param.mvpa.rewrite_mvpa;
rewrite_idx = param.mvpa.rewrite_idx;

% define save path/file for loading bout behavior and fus information
save_fold = param.save_fold;
save_dir = [save_fold 'behavior_fus_region\'];
results_file = [save_dir 'behavior_fus_voxel_' param.behavior.labels '_behav_labels.mat'];

% save_fold = '\\nas6\datastore_brad$\Paulina_test\';
save_dir = [save_fold 'behavior_fus_region\'];
MVPA_fold = [save_dir 'MVPA\'];

results_fold = [MVPA_fold 'results\'];
if ~isfolder(results_fold); mkdir(results_fold); end
% create individual results files for each behavior to check existence thereof
label_meanings = param.behavior.label_meanings;
mvpa_file_flag = zeros(1, size(label_meanings,2));
for i_behavior = 1:size(label_meanings,2)
    mvpa_results_file{i_behavior} = [results_fold label_meanings{i_behavior} '\MVPA_data_' label_meanings{i_behavior} '.mat'];
    if exist(mvpa_results_file{i_behavior}, 'file')
        mvpa_file_flag(i_behavior) = 1;
    end
end


if exist(results_file,'file') && sum(mvpa_file_flag) ~= size(label_meanings,2) ||...
        exist(results_file,'file') && sum(mvpa_file_flag) == size(label_meanings,2) && rewrite_mvpa == 1

    load(results_file, 'I_bout_total', 'I_bout_random_total', 'new_seg');
    
    % save nifti format of segmentation for brain masking
    allen_mask = new_seg; allen_mask(~isnan(allen_mask)) = 1; allen_mask(isnan(allen_mask)) = 0;
    allen_mask_nii = make_nii(allen_mask, [.1 .1 .1]);
    allen_mask_nii_file = [MVPA_fold 'allen_mask.nii'];
    save_nii(allen_mask_nii, allen_mask_nii_file);
    
    % open behavior bout cells into a format easier to work with
    I_bout_total = cat(2,I_bout_total{:});
    I_bout_total_comp = cell(1, size(label_meanings,1));
    for i_behavior = 1:size(label_meanings,2)
        I_bout_total_comp{i_behavior} = cat(3,I_bout_total{i_behavior,:});
    end
    clear I_bout_total
    
    % open random bout cells
    I_bout_random_total = cat(2,I_bout_random_total{:});
    I_bout_random_total_comp = {cat(3,I_bout_random_total{:})};
    clear I_bout_random_total
    
    %%
    % perform behavior-vs-random decoding for all frames and regions
    for i_behavior = 1:size(I_bout_total_comp,2)
        
        fprintf('\n %s: ', label_meanings{i_behavior});
        
        % initialize data structure for saving (per behavior)
        WHOLE_BRAIN = struct;
        REGION = struct;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % extract trials from behavior of interest
        behavior_num = size(I_bout_total_comp{i_behavior},3);
        random_num = size(I_bout_random_total_comp{1},3);
        trial_num = min([behavior_num random_num]);
        
        % save indices for future reproducibility
        idx_file = [results_fold 'bout_indices_' label_meanings{i_behavior} '.mat'];
        if exist(idx_file,'file') && rewrite_idx == 0
            
            load(idx_file)
        
        elseif exist(idx_file,'file') && rewrite_idx == 1 || ~exist(idx_file,'file')
            % select random bout idx's for number of trials needed
            behavior_idx = randperm(behavior_num);
            behavior_idx = behavior_idx(1:trial_num);
            
            random_idx = randperm(random_num);
            random_idx = random_idx(1:trial_num);
            
            % save new idx file
            save(idx_file, 'random_idx', 'behavior_idx');
        end
        
        % extract indices
        trials_behavior = I_bout_total_comp{i_behavior}(:,:,behavior_idx);
        trials_random = I_bout_random_total_comp{1}(:,:,random_idx);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % save nifti format (3D) for each bout and frame
        behav_fold = [MVPA_fold label_meanings{i_behavior} '\'];
        if isfolder(behav_fold); fclose all; rmdir(behav_fold, 's'); end
        mkdir(behav_fold)
        
        clear behav_nii_file random_nii_file
        for i_fr = 1:size(trials_behavior,2)
            fprintf('\n Converting data to NIFTI format; frame # %.0f : trial # ', i_fr);
            
            frame_fold = [behav_fold 'frame_' num2str(i_fr) '\'];
            if ~isfolder(frame_fold); mkdir(frame_fold); end
        
            for i_trial = 1:trial_num
                
                if rem(i_trial,10) == 0
                    fprintf('%0.f ... ', i_trial)
                end
                
                current_behav_volume = reshape(trials_behavior(:, i_fr, i_trial), size(new_seg));
                behav_nii = make_nii(current_behav_volume, [.1 .1 .1]);
                behav_nii_file{i_trial, i_fr} = [frame_fold label_meanings{i_behavior} '_trial_', num2str(i_trial) '.nii'];
                save_nii(behav_nii, behav_nii_file{i_trial, i_fr});
                
                current_random_volume = reshape(trials_random(:, i_fr, i_trial), size(new_seg));
                random_nii = make_nii(current_random_volume, [.1 .1 .1]);
                random_nii_file{i_trial, i_fr} = [frame_fold 'random_trial_', num2str(i_trial) '.nii'];
                save_nii(random_nii, random_nii_file{i_trial, i_fr});
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % perform time-resolved decoding one frame at a time (WHOLE BRAIN)
        fprintf('\n Whole Brain Decoding, frame: ')
        for i_fr = 1:size(trials_behavior,2)
            
            % read nifti images of current frame
            fclose all
            datamvpa{i_fr} = fmri_data([behav_nii_file(:,i_fr); random_nii_file(:,i_fr)]);
            
            % apply allen mask
            datamvpa{i_fr} = apply_mask(datamvpa{i_fr}, allen_mask_nii_file);
            
            % label behav (1) and random (-1) - should always be balanced
            datamvpa{i_fr}.Y = [ones(trial_num,1); -ones(trial_num,1)];
                
            % 10-fold cross validation
            [~, stats_10fold{i_fr}] = predict(datamvpa{i_fr}, 'algorithm_name', 'cv_svm', 'nfolds', 10, 'error_type', 'mcr');
            
            % compute/extract accuracy for each fold
            clear acc_fold
            for i_fold = 1:stats_10fold{i_fr}.cvpartition.NumTestSets
                % test indices from total sample
                idx_test = stats_10fold{i_fr}.teIdx{i_fold};
                idx_test = stats_10fold{i_fr}.Y(idx_test);
                % predicted values of test indices
                val_test = stats_10fold{i_fr}.other_output_cv{i_fold,2};
                % binarize into "+1" and "-1" classes (as in labels "Y")
                val_test(val_test > 0) = 1;
                val_test(val_test < 0) = -1;
                % store accuracy for current fold
                acc_fold(i_fold) = size(find(val_test == idx_test),1)/size(val_test,1);
            end
            
            % ROC analysis
            ROC_10fold{i_fr} = roc_plot(stats_10fold{i_fr}.dist_from_hyperplane_xval, datamvpa{i_fr}.Y == 1, 'threshold', 0, 'nooutput', 'noplot');
            set(gcf,'color','w')
            
            ACC(i_fr) = ROC_10fold{i_fr}.accuracy;
            ACC_fold(:,i_fr) = acc_fold';
            ACC_P(i_fr) = ROC_10fold{i_fr}.accuracy_p;
            
            fprintf('%0.f ', i_fr)
        end
        
        WHOLE_BRAIN.ROC_10fold = ROC_10fold;
        WHOLE_BRAIN.ACC = ACC;
        WHOLE_BRAIN.ACC_fold = ACC_fold;
        WHOLE_BRAIN.ACC_P = ACC_P;
        
        % Bootstrap tests on significant time points before onset for feature weights
        % SIGNIFICANT REGIONS INDICATE PREDICTION???
% % % % % % % % % % % % % % % % % % % % %         ACC_tmp = mafdr(WHOLE_BRAIN.ACC_P,'BHFDR',true);
% % % % % % % % % % % % % % % % % % % % %         sig_fr = find(ACC_tmp(1:25) < 0.05);
% % % % % % % % % % % % % % % % % % % % %         for i_fr = 1:size(sig_fr,2)
% % % % % % % % % % % % % % % % % % % % %             [~, stats_boot{i_fr}] = predict(datamvpa{sig_fr(i_fr)}, 'algorithm_name', 'cv_svm', 'nfolds', 1, 'error_type', 'mcr', 'bootweights', 'bootsamples', 1000);
% % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % %             % correct maps for multiple comparisons
% % % % % % % % % % % % % % % % % % % % %             data_threshold{i_fr} = threshold(stats_boot{i_fr}.weight_obj, .05, 'fdr');
% % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % %             weight_viz = nan(size(data_threshold{i_fr}.removed_voxels));
% % % % % % % % % % % % % % % % % % % % %             weight_viz(data_threshold{i_fr}.removed_voxels == 0) = data_threshold{i_fr}.dat;
% % % % % % % % % % % % % % % % % % % % %             weight_viz = reshape(weight_viz,[80 114 86]);
% % % % % % % % % % % % % % % % % % % % %         end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % perform time-resolved decoding one frame at a time (INDIVIDUAL REGION WEIGHTS)
        fprintf('\n Region-Wise Decoding, frame: ')
        output = struct; % final output for region decoding
        brain = new_seg(:);
        brain = brain(datamvpa{1}.removed_voxels == 0);
        for i_fr = 1:size(trials_behavior,2)
            
            for i_reg = 1:max(new_seg(:))
                
                % also across all folds
                data_fold = []; label_fold = []; clear acc_fold
                for i_fold = 1:stats_10fold{i_fr}.cvpartition.NumTestSets
                    
                    % apply region weights for each fold to testing data of same fold
                    masked_region_weights = stats_10fold{i_fr}.other_output_cv{i_fold,1} .* double(brain == i_reg);
                    
                    % identify measurement indices for the test set of this fold
                    test_idx_fold = stats_10fold{i_fr}.teIdx{i_fold};
                    
                    % apply masked region weights to test measurements
                    test_data_fold = datamvpa{i_fr}.dat(:,test_idx_fold);
                    
                    % compute accuracy for current fold
                    data_tmp = masked_region_weights'*test_data_fold;
                    data_tmp(data_tmp > 0) = 1;
                    data_tmp(data_tmp < 0) = -1;
                    
                    label_tmp = datamvpa{i_fr}.Y(test_idx_fold)';
                    acc_fold(i_fold) = size(find(data_tmp == label_tmp),2)/size(data_tmp,2);
                    
                    % store data and labels for current fold
                    data_fold = [data_fold masked_region_weights'*test_data_fold];
                    label_fold = [label_fold datamvpa{i_fr}.Y(test_idx_fold)'];
                end
                
                % Assess classification performance across all folds
                roc = roc_plot(data_fold(:), label_fold(:) == 1, 'nooutput', 'noplot');
                
                % store per frame and region
                output(i_reg, i_fr).num_vox = sum(double(brain == i_reg));
                output(i_reg, i_fr).acc = roc.accuracy;
                output(i_reg, i_fr).acc_fold = acc_fold(:);
                output(i_reg, i_fr).se = roc.accuracy_se;
                output(i_reg, i_fr).p = roc.accuracy_p;
                
            end
            fprintf('%0.f ', i_fr)
        end 
        
        REGION.output = output;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Save mvpa data for current behavior
        
        % make sure behavior results folder exists
        results_fold_behav = [results_fold label_meanings{i_behavior} '\'];
        if ~isfolder(results_fold_behav); mkdir(results_fold_behav); end
        
        save(mvpa_results_file{i_behavior}, 'WHOLE_BRAIN', 'REGION', '-v7.3');
        
    end
end
    
%% Plot everything

% Results files should all be there no matter what not (no if needed)
for i_behavior = 1:size(label_meanings,2)

    load(mvpa_results_file{i_behavior})
    
    idx_file = [results_fold 'bout_indices_' label_meanings{i_behavior} '.mat'];
    idx = load(idx_file); trial_num = size(idx.behavior_idx,2);
    
    % specify behavior results folder
    results_fold_behav = [results_fold label_meanings{i_behavior} '\'];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % plot whole brain classification accuracy
    f = figure(25); clf; set(gcf,'color','w');
    subplot(10,1,1);
    imagesc(mafdr(WHOLE_BRAIN.ACC_P,'BHFDR',true));
    caxis([0 .05]); colormap gray
    set(gca,'xtick','','ytick','')
    title(['Whole-brain (voxel space) decoding: ' label_meanings{i_behavior} ' (n = ' num2str(trial_num) ')'])

    subplot(10,1,2:10); hold on
    time = -param.behavior.bout_window+1:param.behavior.bout_window;
    time = time * .6; % frames -> sec hardcode

    M = movmean(nanmean(WHOLE_BRAIN.ACC_fold,1),3);
    S = movmean(nanstd(WHOLE_BRAIN.ACC_fold,[],1)./sqrt(size(WHOLE_BRAIN.ACC_fold,1)),3);
    t1 = time; t2 = [t1, fliplr(t1)];
    between = [M + S, fliplr(M - S)];
    fill(t2,between,[65 111 185]/255,'edgecolor','none','facealpha',0.25); hold on
    plot(t1, M, 'color', [65 111 185]/255,'linewidth',1.5);
    plot([0 0], [0 1], 'r')
    plot(time, 0.5*ones(size(time,2),1),'k--')
    set(gca,'ylim',[.4 .85],'tickdir','out');
    ylabel('Accuracy (%)'); xlabel('time (sec)')

    savefig(f, [results_fold_behav 'MVPA_whole_brain_' label_meanings{i_behavior} '.fig']);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % compile region information across time
    for i_fr = 1:size(REGION.output,2)
        % extract accuracy
        reg_acc(:,i_fr) = cell2mat({REGION.output(:,i_fr).acc})'; % from roc
%         reg_acc(:,i_fr) = mean(cat(2,REGION.output(:,i_fr).acc_fold),1)'; % from folds
        % extract p-values
        reg_pval(:,i_fr) = cell2mat({REGION.output(:,i_fr).p})';
    end
    
    f2 = figure(50); clf
    set(gcf,'color','w','position',[680 558 1163 420]);
    subplot(1,2,1);
    imagesc(reg_acc);
    set(gca,'xtick',time(1:5:end)/.6+param.behavior.bout_window,'xticklabel',time(1:5:end)-1)
    title(['Region-wise (voxel space) decoding: ' label_meanings{i_behavior} ' (n = ' num2str(trial_num) ')'])
    ylabel('region #')
    caxis([.5 .8]); colorbar
    
    subplot(1,2,2);
    imagesc(reg_pval);
    set(gca,'xtick',time(1:5:end)/.6+param.behavior.bout_window,'xticklabel',time(1:5:end)-1)
    title('p-values')
    set(gca,'ytick','')
    caxis([0 .05])
    colorbar
    
    savefig(f2, [results_fold_behav 'MVPA_region_wise_' label_meanings{i_behavior} '.fig']);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Predictive regions???

    % A hack to load the region names 
    tmp = load(data_files{2,1});
    [new_seg, new_name] = fus_custom_segment_PW([], tmp.RefAtlas, 0); % load/create segmentation

    % Sort regions by most predictive power before onset
    [Bpre, Ipre] = sort(mean(reg_acc(:,15:25),2),'descend');
    reg_acc_pre = reg_acc(Ipre,:);
    reg_pval_pre = reg_pval(Ipre,:);

    % Sort regions by most predictive power after onset
    [Bpost, Ipost] = sort(mean(reg_acc(:,26:35),2),'descend');
    reg_acc_post = reg_acc(Ipost,:);
    reg_pval_post = reg_pval(Ipost,:);

    f3 = figure(51); clf
    set(gcf,'color','w','position',[680 558 957 420]);
    subplot(1,2,1)
    imagesc(reg_acc_pre);
    title([label_meanings{i_behavior} ' Decoding Sorted: Pre-event'])
    set(gca,'xtick',time(1:5:end)/.6+param.behavior.bout_window,'xticklabel',[time(1:5:end)-1])
    xlabel('time (sec)'); ylabel('region');
    caxis([.5 .8])
    subplot(1,2,2)
    imagesc(reg_acc_post);
    title([label_meanings{i_behavior} ' Decoding Sorted: Post-event'])
    set(gca,'xtick',time(1:5:end)/.6+param.behavior.bout_window,'xticklabel',[time(1:5:end)-1])
    xlabel('time (sec)');
    caxis([.5 .8])

    savefig(f3, [results_fold_behav 'MVPA_region_wise_' label_meanings{i_behavior} '_sorted_pre_post.fig']);

    f4 = figure(52); clf;
    set(gcf,'color','w','position',[680 558 1115 420]);
    for i_reg = 1:5
        % plot individual regions with most predictive power before event onset
        subplot(23,5,i_reg)
        imagesc(mafdr(reg_pval_pre(i_reg,:),'BHFDR',true));
        caxis([0 .05]); colormap gray
        set(gca,'xtick','','ytick','')
        title(new_name{Ipre(i_reg),1})

        subplot(23,5,i_reg+5:5:50); cla; hold on
        % extract x-fold validated time resolved accuracy for current region
        clear reg_acc_tmp
        for i_time = 1:size(REGION.output,2)
            reg_acc_tmp(:,i_time) = REGION.output(Ipre(i_reg),i_time).acc_fold;
        end
        % plot it
        M = movmean(nanmean(reg_acc_tmp,1),3);
        S = movmean(nanstd(reg_acc_tmp,[],1)./sqrt(size(reg_acc_tmp,1)),3);
        t1 = time; t2 = [t1, fliplr(t1)];
        between = [M + S, fliplr(M - S)];
        fill(t2,between,[65 111 185]/255,'edgecolor','none','facealpha',0.25); hold on
        plot(t1, M, 'color', [65 111 185]/255,'linewidth',1.5);
        plot([0 0], [0 1], 'r')
        plot(time, 0.5*ones(size(time,2),1),'k--')
        set(gca,'ylim',[.4 .85],'tickdir','out','xlim',[min(time) max(time)]);
        if i_reg == 1; ylabel(['Pre Acc (%): ' label_meanings{i_behavior}]); end

        % plot individual regions with most predictive power after event onset
        subplot(23,5,i_reg+65)
        imagesc(mafdr(reg_pval_post(i_reg,:),'BHFDR',true));
        caxis([0 .05]); colormap gray
        set(gca,'xtick','','ytick','')
        title(new_name{Ipost(i_reg),1})

        subplot(23,5,[i_reg+5:5:50]+65); hold on
        % extract x-fold validated time resolved accuracy for current region
        clear reg_acc_tmp
        for i_time = 1:size(REGION.output,2)
            reg_acc_tmp(:,i_time) = REGION.output(Ipost(i_reg),i_time).acc_fold;
        end
        % plot it
        M = movmean(nanmean(reg_acc_tmp,1),3);
        S = movmean(nanstd(reg_acc_tmp,[],1)./sqrt(size(reg_acc_tmp,1)),3);
        t1 = time; t2 = [t1, fliplr(t1)];
        between = [M + S, fliplr(M - S)];
        fill(t2,between,[65 111 185]/255,'edgecolor','none','facealpha',0.25); hold on
        plot(t1, M, 'color', [65 111 185]/255,'linewidth',1.5);
        plot([0 0], [0 1], 'r')
        plot(time, 0.5*ones(size(time,2),1),'k--')
        set(gca,'ylim',[.4 .85],'tickdir','out','xlim',[min(time) max(time)]);
        if i_reg == 1; ylabel(['Post Acc (%): ' label_meanings{i_behavior}]); end
        xlabel('time (sec)')

    end

    savefig(f4, [results_fold_behav 'MVPA_region_wise_' label_meanings{i_behavior} '_most_predictive_regions.fig']);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % "Brain map" visualization
    reg_acc_pre_vis = mean(reg_acc(:,15:25),2); % 6 sec before
    reg_acc_pre_vis(reg_acc_pre_vis < .5) = .51;
    reg_acc_post_vis = mean(reg_acc(:,26:35),2); % 6 sec after
    reg_acc_post_vis(reg_acc_post_vis < .5) = .51;

    f5 = figure(53); clf;
    set(gcf, 'color', 'w', 'position',[420 558 1372 420]);

    subplot(1,2,1);
    region_img = fus_assign_region_val(reg_acc_pre_vis, [], new_seg, 0, []);
    imagesc(region_img); axis off
    colormap([.85 .85 .85; parula]); caxis([.5 .7]); colorbar
    title([label_meanings{i_behavior} '"Pre" decoding'])

    subplot(1,2,2);
    region_img = fus_assign_region_val(reg_acc_post_vis, [], new_seg, 0, []);
    imagesc(region_img); axis off
    colormap([.85 .85 .85; parula]); caxis([.5 .7]); colorbar
    title([label_meanings{i_behavior} '"Post" decoding'])

    savefig(f5, [results_fold_behav 'MVPA_region_wise_' label_meanings{i_behavior} '_most_predictive_regions_BRAIN_MAP.fig']);

end
