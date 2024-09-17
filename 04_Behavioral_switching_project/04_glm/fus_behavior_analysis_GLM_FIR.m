function fus_behavior_analysis_GLM(data_files, param)


%% Analysis pipeline for FUS + BEHAVIOR / GLM

% Mace and Gogolla Labs - Max Planck Institute of Neurobiology
% Author: Bradley Edelman and Paulina Wanken
% Date: 11.03.22

rewrite = param.glm_individual_rewrite;

results_dir = 'J:\Paulina Gabriele Wanken\Data\3D\02_GLM\';
results_dir = '\\nas6\datastore_brad$\Paulina_test\';
results_file = [results_dir 'GLM_compiled.mat'];

if ~exist(results_file,'file') || exist(results_file,'file') && rewrite == 1
    %% Analyze and store one session at a time

    % load behavior label meanings
    label_meanings = table2array(readtable('J:/Patrick McCarthy/behaviour_predictions/label_meanings.csv','ReadVariableNames',0));

    regressors_total = cell(size(data_files,1),1)
    list_regressors_total = cell(size(data_files,1),1)
    list_regressors_hrf_total = cell(size(data_files,1),1)
    c_total = cell(size(data_files,1),1)
    D_total = cell(size(data_files,1),1)
    glmInfo_total = cell(size(data_files,1),1)

    for i_session = 1:size(data_files,1) % for each session of all mice % 16 has groom
        i_session

        mouse_names{i_session,:} = data_files{i_session,1}(1, 53:58);
        session_dates{i_session,:}= data_files{i_session,1}(1, 60:67);

        % load fus data
        load(data_files{i_session,1});

        % load behavior labels
        labels = load(data_files{i_session,2}); % load behavior labels
        % add 100 blank frames to the beginning (patrick)
        labels = labels.correct_labels
        labels_behavior = labels

        %% Initializing regressors
        active = zeros(1,size(labels,2));
        egress = zeros(1,size(labels,2));
        groom = zeros(1,size(labels,2));
        inactive = zeros(1,size(labels,2));

        before_a = zeros(1,size(labels,2));
        before_e = zeros(1,size(labels,2));
        before_g = zeros(1,size(labels,2));
        before_i = zeros(1,size(labels,2));

        from_a_to_e = zeros(1,size(labels,2));
        from_a_to_g = zeros(1,size(labels,2));
        from_a_to_i = zeros(1,size(labels,2));

        from_e_to_a = zeros(1,size(labels,2));
        from_e_to_g = zeros(1,size(labels,2));
        from_e_to_i = zeros(1,size(labels,2));

        from_g_to_a = zeros(1,size(labels,2));
        from_g_to_e = zeros(1,size(labels,2));
        from_g_to_i = zeros(1,size(labels,2));

        from_i_to_a = zeros(1,size(labels,2));
        from_i_to_e = zeros(1,size(labels,2));
        from_i_to_g = zeros(1,size(labels,2));

        list_regressors = {'active', 'egress', 'groom', 'inactive', 'before_a', ...
            'before_e', 'before_g', 'before_i', 'from_a_to_e', 'from_a_to_g', ...
            'from_a_to_i', 'from_e_to_a', 'from_e_to_g', 'from_e_to_i', ...
            'from_g_to_a', 'from_g_to_e', 'from_g_to_i', 'from_i_to_a', ...
            'from_i_to_e', 'from_i_to_g'}

        regressors = zeros(20,size(labels,2));

        %% Regressors - each behavior
        active(labels_behavior == 1) = 1;
        egress(labels_behavior == 2) = 1;
        groom(labels_behavior == 3) = 1;
        inactive(labels_behavior == 4) = 1;

        figure (1)
        subplot(2,2,1); plot(active, 'k');
        subplot(2,2,2); plot(egress, 'b');
        subplot(2,2,3); plot(groom, 'r');
        subplot(2,2,4); plot(inactive, 'g');

        regressors (1,:) = active
        regressors (2,:) = egress
        regressors (3,:) = groom
        regressors (4,:) = inactive

        %% Regressors - one frame before each behavior
        before_a = zeros(1,size(labels,2));
        start_active = diff(active)
        start_active(end+1) = 0
        for i = 1:size(labels,2)
            if start_active(1,i) == 1   %diff of active = 1 at switch
                before_a(1,i) = 1
            end
        end

        before_e = zeros(1,size(labels,2));
        start_egress = diff(egress)
        start_egress(end+1) = 0
        for i = 1:size(labels,2)
            if start_egress(1,i) == 1
                before_e(1,i) = 1
            end
        end

        before_g = zeros(1,size(labels,2));
        start_groom = diff(groom)
        start_groom(end+1) = 0
        for i = 1:size(labels,2)
            if start_groom(1,i) == 1
                before_g(1,i) = 1
            end
        end

        before_i = zeros(1,size(labels,2));
        start_inactive = diff(inactive)
        start_inactive(end+1) = 0
        for i = 1:size(labels,2)
            if start_inactive(1,i) == 1
                before_i(1,i) = 1
            end
        end

        figure (2)
        subplot(2,2,1); plot(before_a, 'k');
        hold on; plot(active, 'r');
        subplot(2,2,2); plot(before_e, 'k');
        hold on; plot(egress, 'r');
        subplot(2,2,3); plot(before_g, 'k');
        hold on; plot(groom, 'r');
        subplot(2,2,4); plot(before_i, 'k');
        hold on; plot(inactive, 'r');

        regressors (5,:) = before_a
        regressors (6,:) = before_e
        regressors (7,:) = before_g
        regressors (8,:) = before_i

        %% Regressors - from active to other behaviors
        labels_behavior_odd(labels_behavior == 1) = 25; % numbers are arbitrary (every transition will have unique number)
        labels_behavior_odd(labels_behavior == 2) = 4;
        labels_behavior_odd(labels_behavior == 3) = 2;
        labels_behavior_odd(labels_behavior == 4) = 10;
        labels_behavior_odd(labels_behavior == 0) = 0;

        diff_labels = diff(labels_behavior_odd)
        diff_labels(end+1) = 0

        for i = 2:size(labels,2)
            if diff_labels(1,i-1) == -15
                from_a_to_i(1,i) = 1
            end
        end

        for i = 2:size(labels,2)
            if diff_labels(1,i-1) == -21
                from_a_to_e(1,i) = 1
            end
        end

        for i = 2:size(labels,2)
            if diff_labels(1,i-1) == -23
                from_a_to_g(1,i) = 1
            end
        end


        regressors (9,:) = from_a_to_e
        regressors (10,:) = from_a_to_g
        regressors (11,:) = from_a_to_i

        %% Regressors - from egress to other behaviors
        for i = 2:size(labels,2)
            if diff_labels(1,i-1) == 21
                from_e_to_a(1,i) = 1
            end
        end

        for i = 2:size(labels,2)
            if diff_labels(1,i-1) == -2
                from_e_to_g(1,i) = 1
            end
        end

        for i = 2:size(labels,2)
            if diff_labels(1,i-1) == 6
                from_e_to_i(1,i) = 1
            end
        end

        regressors (12,:) = from_e_to_a
        regressors (13,:) = from_e_to_g
        regressors (14,:) = from_e_to_i

        %% Regressors - from groom to other behaviors

        for i = 2:size(labels,2)
            if diff_labels(1,i-1) == 23
                from_g_to_a(1,i) = 1
            end
        end

        for i = 2:size(labels,2)
            if diff_labels(1,i-1) == 2
                from_g_to_e(1,i) = 1
            end
        end

        for i = 2:size(labels,2)
            if diff_labels(1,i-1) == 8
                from_g_to_i(1,i) = 1
            end
        end

        regressors (15,:) = from_g_to_a
        regressors (16,:) = from_g_to_e
        regressors (17,:) = from_g_to_i

        %% Regressors - from inactive to other behaviors
        for i = 2:size(labels,2)
            if diff_labels(1,i-1) == 15
                from_i_to_a(1,i) = 1
            end
        end

        for i = 2:size(labels,2)
            if diff_labels(1,i-1) == -6
                from_i_to_e(1,i) = 1
            end
        end

        for i = 2:size(labels,2)
            if diff_labels(1,i-1) == -8
                from_i_to_g(1,i) = 1
            end
        end

        regressors (18,:) = from_i_to_a
        regressors (19,:) = from_i_to_e
        regressors (20,:) = from_i_to_g

        %% checking regressors
        figure (3)
        subplot(2,2,1); plot(from_e_to_a, 'k');
        subplot(2,2,2); plot(from_g_to_a, 'b');
        subplot(2,2,3); plot(from_i_to_a, 'r');


        figure (4)
        subplot(2,2,1); plot(from_a_to_e, 'k');
        subplot(2,2,2); plot(from_i_to_e, 'b');
        subplot(2,2,3); plot(from_g_to_e, 'r');

        figure(5)
        plot(from_i_to_a, 'r'); hold on; plot(from_a_to_i, 'k');

        figure(6)
        plot(from_a_to_e, 'r'); hold on; plot(from_e_to_a, 'k');

        rtmp = regressors(1:4,:); % keep only "during" behavior regressors for now
        list_regressors_tmp = list_regressors(1:4);
        list_regressors_tmp(sum(rtmp,2) == 0) = []; % remove "zero regressors" from names
        rtmp(sum(rtmp,2) == 0,:) = []; % remove "zero regressors" from regressor list

        figure(7)
        imagesc(rtmp);
        title('regressors'); ylabel('regressor'); xlabel('time (frames)');
        set(gca,'ytick',1:size(rtmp,1),'yticklabel',list_regressors_tmp)

    % % % % %     list_regressors_old = list_regressors
    % % % % %     rtmp_old = rtmp
    % % % % %     
    % % % % %     for i = size(rtmp_old,1):-1:1 % remove 0 regressors matrix
    % % % % %         if sum(rtmp_old(i,:)) == 0
    % % % % %             rtmp(i,:) = []
    % % % % %             list_regressors{i} = []
    % % % % %         end
    % % % % %     end
    % % % % %     
    % % % % %     %regressors = regressors(1:4,:);
    % % % % %     regressors = rtmp

        regressors_total{i_session} = rtmp
        list_regressors_total{i_session} = list_regressors_tmp

        %% GLM analysis
        % define hrf
        dt_interp = 0.6; % interpolated sampling rate
        [hrf,p] = spm_hrf(dt_interp,[3 16 1 1 20 0 16]);

        % convolve regressors with hrf
        % % pad convolution to avoid boundary issues (add numbers)
        % % regressor matrix needs to be "long" (# regressors x time)
        regressors_hrf = conv2([flipud(rtmp(:,1:50)'); rtmp'; flipud(rtmp(:,end-49:end)')],hrf); % pad convolution to avoid edge effects
        regressors_hrf = regressors_hrf(51:51 + size(regressors,2)-1,:); % keep only same length as data
        regressors_hrf = regressors_hrf(1:size(I_signal,4),:);

        % check design matrix for invertibility
        [D, beta_hat_pre] = fus_check_design(regressors_hrf);

        % define contrast (basic at first)
        c = [1 zeros(1,size(regressors_hrf,2))];
        for i_post = 1:size(regressors_hrf,2)-1
            c = [c; circshift(c(1,:),[0 i_post])];
        end

        % perform GLM analysis
        mask = reshape(points_in, [], 1); % linearize mask
        I_signal_lin = reshape(I_signal, [], size(I_signal,4)); % linearize fus
        glmInfo = fus_glm(I_signal_lin, c, D, beta_hat_pre, double(mask));

        % visualize glm results
        % mask with allen atlas
        allen_mask = reshape(RefAtlas.Regions_crop, 1, []);
        allen_mask(allen_mask == 1) = 0;
        allen_mask(allen_mask > 1) = 1;
        t_stat_viz = glmInfo.t_stat;
        t_stat_viz(~allen_mask,:) = nan;
        t_stat_viz(~mask(:) & allen_mask(:),:) = inf;

        % check manual mask
        mask_viz = reshape(points_in, size(I_signal,1), size(I_signal,2), size(I_signal,3));
        mask_viz = reshape(mask_viz, size(I_signal,1), []);
        mask_viz = fus_2d_to_3d(mask_viz,7);

        mask_allen_viz = reshape(allen_mask, size(I_signal,1), size(I_signal,2), size(I_signal,3));
        mask_allen_viz = reshape(mask_allen_viz, size(I_signal,1), []);
        mask_allen_viz = fus_2d_to_3d(mask_allen_viz,7);

        figure(84); clf
        subplot(1,3,1); imagesc(mask_viz); axis off; title('manual mask')
        subplot(1,3,2); imagesc(mask_allen_viz); axis off; title('allen mask')
        subplot(1,3,3); imagesc(and(mask_viz, mask_allen_viz)); axis off; title('combined mask')

        % reshape into original atlas space
        tmp = reshape(t_stat_viz, size(I_signal,1), size(I_signal,2), size(I_signal,3), size(c,1));
        tmp = reshape(tmp, size(I_signal,1), [], size(c,1));
        tmp = fus_2d_to_3d(tmp,7);
        figure(10); clf
        for i_c = 1:size(c,1)
            subplot(ceil((size(c,1))/4)+1,4,i_c)
            imagesc(tmp(:,:,i_c)); axis off; caxis([-10 10]);
            title(list_regressors_tmp{i_c})
            colormap([0 0 0; parula; .75 .75 .75])
        end

        subplot(ceil((size(c,1))/4)+1,4,ceil((size(c,1))/4)*4+1:ceil((size(c,1))/4)*4+4)
        plot(regressors_hrf + repmat((0:(size(regressors_hrf,2)-1)), size(regressors_hrf,1),1))
        set(gca,'ytick',[0:(size(list_regressors_tmp,2))-1],'yticklabel',list_regressors_tmp,...
            'xlim',[0 size(regressors_hrf,1)], 'fontsize',10,'xtick','','tickdir', 'out')
        xlabel('time')
        drawnow

        list_regressors_hrf_total {i_session} = regressors_hrf;
        c_total {i_session} = c;
        D_total {i_session} = D;
        glmInfo_total {i_session} = glmInfo;


    end

    % save group-level results
    save(results_file, 'list_regressors_hrf_total', 'c_total', 'D_total', 'glmInfo_total', 'data_files', '-v7.3');

end
