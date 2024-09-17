function fus_compile_fus_behavior_data(data_files, param)
% Mace Lab - Max Planck Institute of Neurobiology
% Author: Bradley Edeman
% Date: 05.11.21

%%

mean_behav_during = zeros(134,4)

% extract relevant parameters
rewrite = param.fus_behavior_compile_rewrite;
save_fold = param.save_fold;
bout_window = param.behavior.bout_window;

% define save file for storing behavior bout videos
video_dir = [save_fold 'behavior_bout_videos_4\'];

% define results file for loading session-by-session behavioral fus analysis
results_dir = [save_fold 'behavior_fus_region\'];
results_file = [results_dir 'behavior_fus_region_corrected_behav_labels.mat'];
new_results_file = [results_dir 'behavior_fus_region_corrected_behav_labels_averaged.mat'];

if exist(results_file,'file') && ~exist(new_results_file,'file') ||...
        exist(results_file,'file') && exist(new_results_file,'file') && rewrite == 1
    
    % load behavior label meanings
    label_meanings = param.behavior.label_meanings;
    
    load(results_file) % session-by-session bout info (behavior and fus)
    
    % "open" up cells to concatenate session
    I_bout_total = cat(2,I_bout_total{:});
    I_bout_total_during = cat(2,I_bout_total_during{:});
    B_bout_total = cat(2,B_bout_total{:});
    B_bout_total_info = cat(2,B_bout_total_info{:});
    P_bout_total = cat(2,P_bout_total{:});
    P_bout_total_during = cat(2,P_bout_total_during{:});
    V_bout_total = cat(2,V_bout_total{:});
    V_bout_total_during = cat(2,V_bout_total_during{:})
    V_whole_recording = cat(2,V_whole_recording{:})
    P_whole_recording = cat(2,P_whole_recording{:})
    L_bout_total = cat(2,L_bout_total{:});
    
    % further "open" cells to concatenate bouts
    I_bout_total_comp = cell(1, size(label_meanings,1));
    I_bout_total_during_comp = cell(1, size(label_meanings,1));
    B_bout_total_comp = cell(1, size(label_meanings,1));
    B_bout_total_info_comp = cell(1, size(label_meanings,1));
    P_bout_total_during_comp = cell(1, size(label_meanings,1));
    P_bout_total_comp = cell(1, size(label_meanings,1));
    V_bout_total_during_comp = cell(1, size(label_meanings,1));
    V_bout_total_comp = cell(1, size(label_meanings,1));
    L_bout_total_comp = cell(1, size(label_meanings,1));
    L_bout_behav_before = cell (size(label_meanings,1),1)
    L_label_sum = cell (size(label_meanings,1),1)
    L_label_probability = cell (size(label_meanings,1),1)
    for i_behavior = 1:size(label_meanings,1)
        I_bout_total_comp{i_behavior} = cat(3,I_bout_total{i_behavior,:});
        I_bout_total_during_comp{i_behavior} = cat(2,I_bout_total_during{i_behavior,:});
        B_bout_total_comp{i_behavior} = cat(2,B_bout_total{i_behavior,:});
        B_bout_total_info_comp{i_behavior} = cat(2,B_bout_total_info{i_behavior,:});
        P_bout_total_during_comp{i_behavior} = cat(2,P_bout_total_during{i_behavior,:})
        P_bout_total_comp{i_behavior} = cat(2,P_bout_total{i_behavior,:});
        V_bout_total_during_comp{i_behavior} = cat(2,V_bout_total_during{i_behavior,:})
        V_bout_total_comp{i_behavior} = cat(2,V_bout_total{i_behavior,:});
        L_bout_total_comp{i_behavior} = cat(2,L_bout_total{i_behavior,:})
    end
    
    % for each behavior, check if a bout review file (excel) exists
    % % if it does, load it and remove bouts that are not approved
    %     bout_approval = cell(1, size(label_meanings,1));
    %     for i_behavior = 1:size(label_meanings,1)
    %
    %         review_file = [video_dir label_meanings{i_behavior} '\Bout_param_review_onset_5_sec_frames.xlsx'];
    %         if exist(review_file,'file')
    %
    %             [~, TXT, RAW] = xlsread(review_file); % load review file
    %             approve = RAW(2:end,6); % look only at last column where approval will be marked as y/n
    %             approve(cellfun(@(v)isnan(v), approve)) = {1}; % any blank responses get included (1's)
    %             approve(cellfun(@(v)strcmp(v,'y'), approve)) = {1}; % 'y' gets included (1's)
    %             approve(cellfun(@(v)strcmp(v,'n'), approve)) = {0}; % 'n' gets removed (0's)
    %
    %             bout_approval{i_behavior} = approve; % store and use later when compiling bout results
    %
    %         else
    %
    %             % if review file doesnt exist, approve all bouts
    %             bout_approval{i_behavior} = ones(size(B_bout_total_info_comp{i_behavior},2),1);
    %
    %         end
    %
    %         % remove unapproved bouts from all compiled info
    %         I_bout_total_comp{i_behavior}(:,:,bout_approval{i_behavior} == 0) = [];
    %         I_bout_total_during_comp{i_behavior}(bout_approval{i_behavior} == 0) = [];
    %         B_bout_total_comp{i_behavior}(:,bout_approval{i_behavior} == 0) = [];
    %         B_bout_total_info_comp{i_behavior}(bout_approval{i_behavior} == 0) = [];
    %         P_bout_total_comp{i_behavior}(:,bout_approval{i_behavior} == 0) = [];
    %         P_bout_total_during_comp{i_behavior}(bout_approval{i_behavior} == 0) = [];
    %         V_bout_total_comp{i_behavior}(:,bout_approval{i_behavior} == 0) = [];
    %         V_bout_total_during_comp{i_behavior}(bout_approval{i_behavior} == 0) = [];
    %         L_bout_total_comp{i_behavior}(:,bout_approval{i_behavior} == 0) = [];
    %
    %     end
    
    %% Compile results across sessions and plot
    close all
    frame_window = 25;
    time_in_sec = ([1:50]-25)*0.6
    
    %% create color code for region activity plotting - this will be used for various plots
    big_reg = unique(new_name(:,2),'stable');
    cmap1 = flipud(colorcube(size(big_reg,1)));
    C = zeros(size(new_name,1),3);
    for i_bigR = 1:size(big_reg,1)
        reg_idx = find(strcmp(new_name(:,2),big_reg{i_bigR}));
        C(reg_idx,:) = repmat(cmap1(i_bigR,:),size(reg_idx,1),1);
        N(reg_idx) = i_bigR;
    end
    
    %% Plot regions x time for all behaviors
    figure(1); clf;
    % randomly select x number of bouts (B and PW 20211511)
    minbout = min(cellfun(@(v)size(v,3), I_bout_total_comp)); % find min number of bouts across all behaviors
    
    subplot(12,21,1:21:8*21);
    imagesc(N'); set(gca,'colormap',cmap1); axis off
    for i_behavior = 1:size(label_meanings,1)
        
        % randomly select min bout indices for each behavior
        rand_idx(i_behavior,:) = datasample(1:size(I_bout_total_comp{i_behavior},3), minbout, 2, 'replace', false);
        %uncomment if you want to select new rand_idx
        sort(rand_idx,'ascend')
        
        %zscoring to baseline
        ave_behav = nanmean(I_bout_total_comp{i_behavior}(:,:,rand_idx(i_behavior,:)),3);
        ave_behav_z  =(ave_behav-mean(ave_behav(:,1:20),2))./std(ave_behav(:,1:20),[],2); % z-scoring to baseline (before onset)
        %subplot(1,4,i_behavior); imagesc(ave_behav_z); % plotting activity around onset, z-scored to baseline before
        
        subplot(12,21,[2:21:8*21, 3:21:8*21, 4:21:8*21, 5:21:8*21, 6:21:8*21] + 5*(i_behavior-1));
        imagesc(ave_behav_z);
        %not zscored %imagesc(movmean(nanmean(I_bout_total_comp{i_behavior}(:,:,rand_idx(i_behavior,:)),3),2,2));
        title([label_meanings{i_behavior} ': n = ' num2str(size(I_bout_total_comp{i_behavior}(rand_idx(i_behavior,:)),2))])
        %             set(gca,'xtick',1:10:frame_window*2,'ytick','',...
        %                     'xticklabel',-frame_window:10:frame_window)
        set(gca,'xtick',1:10:frame_window*2,'ytick','',...
            'xticklabel',-12:6:12)
        caxis([-7 7]);
        
        subplot(12,21,[191:21:10*21, 192:21:10*21, 193:21:10*21, 194:21:10*21, 195:21:10*21] + 5*(i_behavior-1));
        t1 = -frame_window+1:frame_window;
        t2 = [t1, fliplr(t1)];
        
        P_bout = P_bout_total_comp{i_behavior}(:,rand_idx(i_behavior,:));
        M = nanmean(P_bout,2)';
        M = M - nanmean(M(1:19));
        S = nanstd(P_bout,[],2)';
        
        if isempty(M) || isempty(S)
            M = zeros(size(t1,2),1);
            S = zeros(size(t1,2),1);
        end
        
%         % shaded std
%         between = [M + S, fliplr(M - S)];
%         
%         plot(t1, M, 'k'); hold on
%         fill(t2, between, [.65 .65 .65], 'edgecolor', 'none', 'facealpha', 0.5)
%         set(gca,'xtick',1:10:frame_window*2,'ytick','',...
%             'xticklabel',-frame_window:10:frame_window)
%         title(label_meanings{i_behavior});
%         set(gca,'xtick',1:10:frame_window*2,'ytick','',...
%             'xticklabel',-12:6:12)
%         xlabel('time (frames)');
    end
    
    %% Plot "probability of behavior" for window around bout onset for all behaviors
    %     figure(2); clf;
    %     for i_behavior = 1:size(label_meanings,1)
    %         subplot(2,4,i_behavior); hold on
    %
    %          % time vectors for plotting
    %         t1 = -bout_window+1:bout_window;
    %         t2 = [t1, fliplr(t1)];
    %
    %         B_bout = B_bout_total_comp{i_behavior};
    %
    %         M = nanmean(B_bout,2)'; % mean
    %         S = nanstd(B_bout,[],2)'./sqrt(size(B_bout,2)); % sem
    %         % shaded std
    %         between = [M + S, fliplr(M - S)];
    %
    %         plot(t1, M, 'k');
    %         fill(t2, between', [.65 .65 .65], 'edgecolor', 'none', 'facealpha', 0.5)
    %         title([label_meanings{i_behavior} ': n = ' num2str(size(B_bout_total_comp{i_behavior},2))])
    %         xlabel('time (frames)'); ylabel('Behavior Prob.')
    %         grid minor
    %
    %         % also plot histogram of bout length
    %         subplot(2,4,i_behavior +  size(label_meanings,1))
    %         B_bout_info_behav_tmp = cat(2,B_bout_total_info_comp{i_behavior}(:,rand_idx(i_behavior,:)));
    %         B_bout_info_behav{i_behavior} = cellfun(@(v)size(v,2),B_bout_info_behav_tmp);
    %         hist(B_bout_info_behav{i_behavior},0.5:1:30)
    %         set(gca,'xlim',[0 30],'ylim',[0 50])
    %         ylabel('count'); xlabel('bout length (frames)')
    %
    %     end
    
    %% plot probability of all behaviors around each behavior
    
    figure(3)
    for i_behavior = 1:size(label_meanings,1)
        
        for  i_frame = 1:size( L_bout_total_comp{i_behavior},1)
            for i_label = 1:size(label_meanings,1)
                
                % also take same random trials for this (saved in the "rand_idx" variable)
                L_bout_frame {i_behavior,i_label} (i_frame) = sum( L_bout_total_comp{i_behavior}(i_frame,rand_idx(i_behavior,:)) == i_label) % number of labels of each behavior for frame 0-40
                L_bout_probability_frame {i_behavior,i_label} (i_frame)  = (sum( L_bout_total_comp{i_behavior}(i_frame,rand_idx(i_behavior,:)) == i_label))/(size(rand_idx,2)) % number of labels of each behavior divided by amount of trials/bouts
                
            end
        end
        
        subplot(1,4,i_behavior); hold on
        plot(time_in_sec, L_bout_probability_frame{i_behavior,1},'Color', '#77AC30'); hold on
        plot(time_in_sec, L_bout_probability_frame{i_behavior,2}, 'r'); hold on
        plot(time_in_sec, L_bout_probability_frame{i_behavior,3}, 'b'); hold on
        plot(time_in_sec,L_bout_probability_frame{i_behavior,4}, 'k'); hold on
        
        title([label_meanings{i_behavior} ': n = ' num2str(size(I_bout_total_comp{i_behavior}(rand_idx(i_behavior,:)),2))])
        xlabel('time (frames)'); ylabel('Behavior Prob.')
        grid minor
        set(gca,'xtick',1:10:frame_window*2,'ytick','',...
            'xticklabel',-12:6:12)
        labels = {'active', 'egress', 'groom', 'inactive'}
        
        if i_behavior == 4
            lgd = legend(labels, 'Location', 'northeast');
        end
    end
    
    %% Illustrate mean activation during bout (independent of bout length) for all behaviors
    figure(4); clf
    for i_behavior = 1:size(label_meanings,1)
        
        % find average brain activity for each bout
        % % only look at the balanced and random trial numbers
        for i_bout = 1:size(rand_idx,2)
            I_bout_during_ave{i_behavior}(:,i_bout) = nanmean(I_bout_total_during_comp{i_behavior}{rand_idx(i_behavior,i_bout)},2);
        end
        
        subplot(2,2,i_behavior);
        region_img = fus_assign_region_val((nanmean(I_bout_during_ave{i_behavior},2)), [], new_seg, 0, []);
        imagesc(region_img); axis off; caxis([-1.5 1.5])
        title(['During ' label_meanings{i_behavior} ': n = ' num2str(size(I_bout_total_during_comp{i_behavior}(rand_idx(i_behavior,:)),2))]);
    end
    colormap([.65 .65 .65; fireice])
    
    %% Mean activation for 8 frames (~ 5 sec) preceding bout onset
    figure(5); clf
    for i_behavior = 1:size(label_meanings,1)
        
        ave_behav  = nanmean(I_bout_total_comp{i_behavior}(:,:,rand_idx(i_behavior,:)),3);
        mean_behav_before(:,i_behavior) = nanmean(ave_behav(:,bout_window - 5:bout_window-1),2);
        
        region_img = fus_assign_region_val(mean_behav_before(:,i_behavior), [], new_seg, 0, []);
        
        subplot(2,2,i_behavior);
        imagesc(region_img); axis off; caxis([-0.25 0.25]);
        title(['Before ' label_meanings{i_behavior} ': n = ' num2str(size(I_bout_total_comp{i_behavior}(rand_idx(i_behavior,:)),2))])
    end
    colormap([.85 .85 .85; fireice])
    
    %% Mean activation for 8 frame (~5 sec) after bout onset
    figure(6); clf
    for i_behavior = 1:size(label_meanings,1)
        
        ave_behav = nanmean(I_bout_total_comp{i_behavior}(:,:,rand_idx(i_behavior,:)),3);
        mean_behav_after(:,i_behavior) = nanmean(ave_behav(:,bout_window:bout_window + 8),2);
        
        region_img = fus_assign_region_val(mean_behav_after(:,i_behavior), [], new_seg, 0, []);
        
        subplot(2,2,i_behavior);
        imagesc(region_img); axis off; caxis([-1 1]);
        title(['After ' label_meanings{i_behavior} ': n = ' num2str(size(I_bout_total_comp{i_behavior}(rand_idx(i_behavior,:)),2))])
        %caxis([-1.5 1.5])
    end
    colormap([.85 .85 .85; fireice])
    
    
    %% plot pairwise activity combinations during behavior
    combos = combntns(1:size(label_meanings,1),2);
    figure(7); clf
    for i_combo = 1:size(combos,1)
        
        pair = combos(i_combo,:);
        
        % These are already the pre-selected and balanced bouts
        act1 = nanmean(I_bout_during_ave{pair(1)},2);
        act2 = nanmean(I_bout_during_ave{pair(2)},2);
        
        subplot(2, size(combos,1)/2, i_combo)
        
        scatter(act1, act2, 15, C, 'filled'); hold on % use colors from earlier to designate regions
        plot([-5 5], [-5 5],'color','k','linewidth',1.5)
        plot([-5 5], [0 0],'color','k','linewidth',1.5)
        plot([0 0], [-1 1],'color','k','linewidth',1.5)
        set(gca,'xlim',[-.5 .5],'ylim',[-.5 .5],'color',[.7 .7 .7])
        title([label_meanings{pair(1)} ' (x) vs ' label_meanings{pair(2)} ' (y)'])
        grid minor
    end
    subplot(2, size(combos,1)/2, 1)
    xlabel('delta I/I'); ylabel('delta I/I')
    subplot(2, size(combos,1)/2, 5); xlabel('During')
    
    %% Plot brain regions (thresholded) for each quadrant DURING (ADDED 26.11.21)
    
    % specify which pair
    combo_idx = 4; % egress vs groom
    pair = combos(combo_idx,:);
    
    % These are already the pre-selected and balanced bouts
    act1 = nanmean(I_bout_during_ave{pair(1)},2);
    act2 = nanmean(I_bout_during_ave{pair(2)},2);
    
    for i=1:size(mean_behav_after,2)
        mean_behav_during (:,i) = nanmean(I_bout_during_ave{1,i},2);
    end
    
    % obtain brain outline for plotting
    outline = reshape(new_seg, size(new_seg,1), []);
    outline = fus_2d_to_3d(outline,7);
    
    % threshold the activation
    act_thresh = 0.1;
    
    % rules define quadrants (ex: first rule is negative for var 1 and positive for var 2)
    % % we also threshold this according to the activation threshold above
    regions{1} = nan(size(N,2),1); regions{1}(act1 < -act_thresh & act2 > act_thresh) = N(act1 < -act_thresh & act2 > act_thresh);
    regions{2} = nan(size(N,2),1); regions{2}(act1 > act_thresh & act2 > act_thresh) = N(act1 > act_thresh & act2 > act_thresh);
    regions{3} = nan(size(N,2),1); regions{3}(act1 < -act_thresh & act2 < -act_thresh) = N(act1 < -act_thresh & act2 < -act_thresh);
    regions{4} = nan(size(N,2),1); regions{4}(act1 > act_thresh & act2 < -act_thresh) = N(act1 > act_thresh & act2 < -act_thresh);
    
    rlabels = {'-/+', '+/+', '-/-', '+/-'};
    figure(25); clf
    for i_quad = 1:4
        subplot(2,2,i_quad);
        
        region_img = fus_assign_region_val(regions{i_quad}, [], new_seg, 0, []);
        region_img(isnan(region_img) & ~isnan(outline)) = 0;
        cmap2 = [.65 .65 .65; .85 .85 .85; cmap1];
        imagesc(region_img); colormap(cmap2); caxis([-1 range(N)+1]); axis off
        title(['DURING ' label_meanings{pair(1)} '/' label_meanings{pair(2)} ' : ' rlabels{i_quad}])
    end
    
    %% plot pairwise activity combinations before behavior
    combos = combntns(1:size(label_meanings,1),2);
    figure(8); clf
    for i_combo = 1:size(combos,1)
        
        pair = combos(i_combo,:);
        
        act1 = mean_behav_before(:,pair(1));
        act2 = mean_behav_before(:,pair(2));
        
        subplot(2, size(combos,1)/2, i_combo)
        
        scatter(act1, act2, 15, C, 'filled'); hold on
        plot([-5 5], [-5 5],'color','k','linewidth',1.5)
        plot([-5 5], [0 0],'color','k','linewidth',1.5)
        plot([0 0], [-1 1],'color','k','linewidth',1.5)
        set(gca,'xlim',[-.3 .3],'ylim',[-.3 .3],'color',[.7 .7 .7])
        title([label_meanings{pair(1)} ' (x) vs ' label_meanings{pair(2)} ' (y)'])
        grid minor
    end
    subplot(2, size(combos,1)/2, 1)
    xlabel('delta I/I'); ylabel('delta I/I')
    subplot(2, size(combos,1)/2, 5); xlabel('Before')
    
    %% Plot brain regions (thresholded) for each quadrant before (ADDED 26.11.21)
    
    % specify which pair
    combo_idx = 4; % egress vs groom
    pair = combos(combo_idx,:);
    
    % These are already the pre-selected and balanced bouts
    act1 = mean_behav_before(:,pair(1));
    act2 = mean_behav_before(:,pair(2));
    
    % obtain brain outline for plotting
    outline = reshape(new_seg, size(new_seg,1), []);
    outline = fus_2d_to_3d(outline,7);
    
    % threshold the activation
    act_thresh = 0.05;
    
    % rules define quadrants (ex: first rule is negative for var 1 and positive for var 2)
    % % we also threshold this according to the activation threshold above
    regions{1} = nan(size(N,2),1); regions{1}(act1 < -act_thresh & act2 > act_thresh) = N(act1 < -act_thresh & act2 > act_thresh);
    regions{2} = nan(size(N,2),1); regions{2}(act1 > act_thresh & act2 > act_thresh) = N(act1 > act_thresh & act2 > act_thresh);
    regions{3} = nan(size(N,2),1); regions{3}(act1 < -act_thresh & act2 < -act_thresh) = N(act1 < -act_thresh & act2 < -act_thresh);
    regions{4} = nan(size(N,2),1); regions{4}(act1 > act_thresh & act2 < -act_thresh) = N(act1 > act_thresh & act2 < -act_thresh);
    
    rlabels = {'-/+', '+/+', '-/-', '+/-'};
    figure(9); clf
    for i_quad = 1:4
        subplot(2,2,i_quad);
        
        region_img = fus_assign_region_val(regions{i_quad}, [], new_seg, 0, []);
        region_img(isnan(region_img) & ~isnan(outline)) = 0;
        cmap2 = [.65 .65 .65; .85 .85 .85; cmap1];
        imagesc(region_img); colormap(cmap2); caxis([-1 range(N)+1]); axis off
        title(['BEFORE ' label_meanings{pair(1)} '/' label_meanings{pair(2)} ' : ' rlabels{i_quad}])
    end
    
    for i = 1:size(regions,2)
        fprintf('\nRegions for condition %s:\n', num2str(i))
        nonnan_idx = find(~isnan(regions{i}));
        for ii = 1:size(nonnan_idx,1)
            fprintf('%s: %s\n', new_name{nonnan_idx(ii),1}, new_name{nonnan_idx(ii),2});
        end
        fprintf('\n')
    end
    
%     %%  plot box plot of pupil diameter before, during and after behavior
%     figure(10); clf
%     
%     for i_behavior = 1:size(label_meanings,1)
%         bouts_before=zeros(size(P_bout_total_comp{1,i_behavior}(:,rand_idx(i_behavior,:)),2),1);
%         bouts_after=zeros(size(P_bout_total_comp{1,i_behavior}(:,rand_idx(i_behavior,:)),2),1);
%         
%         for i_bout=1:size(P_bout_total_comp{1,i_behavior}(:,rand_idx(i_behavior,:)),2)
%             i_bout
%             
%             % keep random and balanced trials
%             P_bout_during_ave{i_behavior}(i_bout) = mean(P_bout_total_during_comp{i_behavior}{rand_idx(i_behavior,i_bout)});
%             
%             before = frame_window - 19:frame_window;
%             after = frame_window:frame_window+19;
%             bout = P_bout_total_comp{1,i_behavior}(:,rand_idx(i_behavior,i_bout));
%             bout = bout - nanmean(bout(1:19));
%             
%             mean_before = mean(bout(before,:));
%             bouts_before(i_bout) =(mean_before)
%             mean_after = mean(bout(after,:));
%             bouts_after(i_bout) =(mean_after)
%             
%         end
%         
%         subplot(1,4,i_behavior);
%         boxplot([bouts_before(:) bouts_after(:)])
%         set(gca,'xticklabel',{'before', 'after'})
%         title([' pupil ' ]);
%         %         subtitle ([' before ' label_meanings{i_behavior}])
%         ylim([-150 150])
%         drawnow
%         
%     end
    %% plot time-locked VBA trace for each behavior
    figure(11); clf
    for i_behavior = 1:size(label_meanings,1)
        
        subplot(1,size(label_meanings,1),i_behavior); hold on
        t1 = -frame_window+1:frame_window;
        t2 = [t1, fliplr(t1)];
        
        % keep random and balanced trials
        V_bout = V_bout_total_comp{i_behavior}(:,rand_idx(i_behavior,:));
        M = nanmean(V_bout,2)';
        M = M - nanmean(M(1:19))
        S = nanstd(V_bout,[],2)';
        
        if isempty(M) || isempty(S)
            M = zeros(size(t1,2),1);
            S = zeros(size(t1,2),1);
        end
        
%         % shaded std
%         between = [M + S, fliplr(M - S)];
%         
%         plot(t1, M, 'k');
%         fill(t2, between, [.75 .75 .75], 'edgecolor', 'none', 'facealpha', 0.5)
%         title(label_meanings{i_behavior});
%         xlabel('time (frames)'); ylabel('VBA position')
%         set(gca,'xtick',1:10:frame_window*2,'ytick','',...
%             'xticklabel',-12:6:12)
%         ylim([-15 40])
    end
    
    %% kmeans clustering
    
    % keep random and balanced trials
    ave_behav_groom = nanmean(I_bout_total_comp{3}(:,:,rand_idx(3,:)),3);
    ave_behav_egress = nanmean(I_bout_total_comp{2}(:,:,rand_idx(2,:)),3);
    %M=cat(2,zscore(ave_behav_egress,[],2),zscore(ave_behav_groom,[],2));
    %M=zscore(ave_behav_groom,[],2);
    
    M1=(ave_behav_groom-mean(ave_behav_groom(:,1:20),2))./std(ave_behav_groom(:,1:20),[],2); %correct for different baselines
    M2=(ave_behav_egress-mean(ave_behav_egress(:,1:20),2))./std(ave_behav_egress(:,1:20),[],2);
    M=cat(2,M1,M2); %plot M and do clusters on M if you want to cluster for both behaviors at the same time
    
    nclus=6;
    clus1=kmeans(M1,nclus);                  % will give you 8 clusters based on timeseries
    [clusord,ind]=sort(clus1,'ascend');
    
    clus2=kmeans(M2,nclus);                  % will give you 8 clusters based on timeseries
    [clusord,ind]=sort(clus2,'ascend');
    
    %% plots clusters separated by lines
    figure(12);
    subplot(1,2,1)
    imagesc(M1(ind,:));
    caxis([-5 5]);
    hold on
    for i=1:nclus-1
        indline=find(clusord>i,1,'first');
        plot([0 80],[indline indline],'k');
        title([ ' groom kmeans clusters ']);
        
    end
    
    subplot(1,2,2)
    imagesc(M2(ind,:));
    caxis([-5 5]);
    hold on
    for i=1:nclus-1
        indline=find(clusord>i,1,'first');
        plot([0 80],[indline indline],'k');
        title([ ' egress kmeans clusters ']);
        
    end
    
    %% plots mean activity of clusters for groom and egress
    figure(13);
    for i=1:nclus
        col=lines(nclus);
        subplot(nclus,1,i);
        plot(mean(M1(clus1==i,:)),'Color',col(i,:));
        title([ ' mean activity of groom clusters ']);
    end
    
    figure(14);
    for i=1:nclus
        col=lines(nclus);
        subplot(nclus,1,i);
        plot(mean(M2(clus2==i,:)),'Color',col(i,:));
        title([ ' mean activity of egress clusters ']);
    end
    
    %% plots clusters on brain segmentation for groom
    region_brain=zeros(size(new_seg));
    for j=1:nclus
        list=find(clus1==j);
        for i=1:length(list)
            region_brain(new_seg==list(i))=j;
        end
    end
    
    figure(15); montage(region_brain,'DisplayRange', [0 nclus]);
    colormap parula
    title([ ' clusters groom']);
    
    %% plots clusters on brain segmentation for egress
    region_brain=zeros(size(new_seg));
    for j=1:nclus
        list=find(clus2==j);
        for i=1:length(list)
            region_brain(new_seg==list(i))=j;
        end
    end
    
    figure(16); montage(region_brain,'DisplayRange', [0 nclus]);
    colormap parula
    title([ ' clusters egress ']);
    
    %% ranking brain regions by mean activity during behavior
    
    win=20:30; %test different windows, before, during..
    ranking=mean(M1(:,win),2); %ranking groom activity
    [rankord,ind]=sort(ranking,'descend');
    
    ranking2=mean(M2(:,win),2); %ranking egress activity
    [rankord,ind2]=sort(ranking2,'descend');
    
    %% groom
    figure(17); imagesc(M1(ind,:));
    
    for i=1:134
        region_brain(new_seg==i)=ranking(i);
    end
    title([ ' groom ']);
    subtitle ([' sorted by max activity '])
    set(gca,'xtick',1:10:frame_window*2,'ytick','',...
        'xticklabel',-12:6:12)
    
    %figure; montage(region_brain,'DisplayRange', [-8 8]);
    
    
    %% plotting both behaviors sorted by activity during one behavior
    figure (18); subplot(1,2,1); imagesc(M1(ind,:)); %caxis([-10 10]);
    title([ ' groom ']);
    subtitle ([' sorted by max activity '])
    subplot(1,2,2); imagesc(M2(ind,:)); %caxis([-8 8]);
    title([ ' egress ']);
    set(gca,'xtick',1:10:frame_window*2,'ytick','',...
        'xticklabel',-12:6:12)
    
    figure (19); subplot(1,2,1); imagesc(M2(ind2,:)); %caxis([-8 8]);
    title([ ' egress ']);
    subtitle ([' sorted by max activity '])
    subplot(1,2,2); imagesc(M1(ind2,:)); %caxis([-8 8]);
    title([ ' groom ']);
    set(gca,'xtick',1:10:frame_window*2,'ytick','',...
        'xticklabel',-12:6:12)
    
    drawnow
    
    %% VBA quantification
    list_mice = {'PWWT32' , 'PWWT33', 'PWWT34', 'PWWT37', 'PWWT38', 'PWWT40', 'PWWT41', 'PWWT43','PWWT44','PWWT45','PWWT46'};
    
    for i_mouse=1:size(V_whole_recording,2) % find sessions belonging to same mouse
        mouse_number(i_mouse)=find(contains(list_mice,mouse_names{i_mouse}));
    end
    
    mouse_first_sessions=[1 find(diff(mouse_number))+1]; % find first session for plotting lines
    
    %% plot heatmap of VBA for all mice
    
    figure(20); clf;
    imagesc(V_whole_recording');
    caxis([20 60])
    hold on;
    for i=1:max(mouse_number) %plotting lines on top of heatmap
        plot([1 size(V_whole_recording,1)], [mouse_first_sessions(i) mouse_first_sessions(i) ]-0.5,'w');
        
    end
    
    %% plot heatmap of pupil for all mice
    
    figure(21); clf;
    imagesc(P_whole_recording');
    hold on;
    for i=1:max(mouse_number) %plotting lines on top of heatmap
        plot([1 size(P_whole_recording,1)], [mouse_first_sessions(i) mouse_first_sessions(i) ]-0.5,'w');
    end
    
    %% plot heatmap of VBA for one mouse
    figure(22); clf
    
    mouse = 'PWWT34'
    session_number = count(mouse_names, mouse)
    session_number = find(session_number == 1)
    
    for i_v = 1:size(session_number,1) %%%HIER STIMMT ETWAS NICHT
        
        vba_animal{i_v} = V_whole_recording(:,session_number(1)+i_v);
        vba_animal1 = cat(2,vba_animal{:})
        imagesc(vba_animal1');
        
    end
    
    hold on;
    for i=1:size(session_number,1)
        plot([1 size(V_whole_recording,1)], [i  i]-0.5,'w');
    end
    
    %% bar chart of % of time spent outside
    figure(23)
    
    V_above45=cell(60,1)
    V_above_size=cell(60,1)
    
    for i_v = 1:size(V_whole_recording,2)
        V_above45{i_v,1} = find(V_whole_recording(:,i_v) > 47) ;
        V_above_size{i_v,1} = size(V_above45{i_v,1},1)
        
        V_above_percent{i_v,1}=V_above_size{i_v,1}/1025
        V_above = cat(1,V_above_percent{:})
        bar(V_above)
        title(['percentage of time spent outside of all sessions'])
    end
    
    %% compile bout behavior data across all sessions and behaviors, and plot regions x time
    frame_window=25
    figure(24); clf;
    for i_behavior = 1:size(label_meanings,1)
        %B_bout_behav{i_behavior} = cat(2,B_bout_total_comp{i_behavior,:});
        subplot(2,4,i_behavior); hold on
        
        % time vectors for plotting
        t1 = -frame_window+1:frame_window;
        t2 = [t1, fliplr(t1)];
        
        B_bout = B_bout_total_comp{i_behavior};
        
        M = nanmean(B_bout,2)'; % mean
        S = nanstd(B_bout,[],2)'./sqrt(size(B_bout,2)); % sem
        % shaded std
%         between = [M + S, fliplr(M - S)];
%         
%         plot(t1, M, 'k');
%         fill(t2, between', [.65 .65 .65], 'edgecolor', 'none', 'facealpha', 0.5)
%         title([label_meanings{i_behavior} ': n = ' num2str(size(B_bout_total_comp{i_behavior},2))])
%         xlabel('time (frames)'); ylabel('Behavior Prob.')
%         grid minor
        
        % also plot histogram of bout length
        
        subplot(2,4,i_behavior +  size(label_meanings,1))
        %                 B_bout_info_behav_tmp = cat(2,B_bout_total_info_comp{i_behavior,:});
        %                 B_bout_info_behav{i_behavior} = cellfun(@(v)size(v,2),B_bout_info_behav_tmp);
        %B_bout_info_behav_tmp = cat(2,B_bout_total_info_comp{i_behavior,:});
        B_bout_info_behav{i_behavior} = cellfun(@(v)size(v,2),B_bout_total_info_comp);
        hist(B_bout_info_behav{i_behavior},0.5:1:15)
        %set(gca,'xlim',[0 500],'ylim',[0 30])
        ylabel('count'); xlabel('bout length (frames)')
        
    end
end

%% Save compiled results
save(new_results_file, 'I_bout_total_comp', 'I_bout_total_during_comp','P_bout_total_comp',...
    'P_bout_total_during_comp', 'B_bout_total_comp', 'B_bout_total_info_comp', ...
    'V_bout_total_comp','V_bout_total_during_comp', 'rand_idx', 'V_whole_recording', ...
    'P_whole_recording', 'L_bout_total' ,'new_name', 'new_seg', 'rand_idx', '-v7.3');

%save(x, 'mean_behav_before', 'mean_behav_during', 'label_meanings', '-v7.3');
end

