function fus_plot_behavior_bout_info(new_name, I_bout_session, B_bout_session, param)
% Mace and Gogolla Labs - Max Planck Institute of Neurobiology
% Author: Bradley Edeman
% Date: 03.11.21

%%
if param.plot_session == 1
    
    bout_window = param.bout_window;

    % load behavior label meanings
    label_meanings = table2array(readtable('J:/Patrick McCarthy/behaviour_predictions/label_meanings.csv','ReadVariableNames',0));

    % find and plot "big" region classes
    big_reg = unique(new_name(:,2));
    for i_big_reg = 1:size(big_reg,1)
        big_regC(strcmp(new_name(:,2), big_reg(i_big_reg))) = i_big_reg;
    end
    cmap = flipud(colorcube(size(big_regC,2)));

    % plot time-locked brain region activity for each behavior
    figure(40); clf
    subplot(1,21,1);
    imagesc(big_regC'); set(gca,'colormap', cmap); axis off
    for i_behavior = 1:size(label_meanings,1)
        subplot(1,21,[2:6] + 5*(i_behavior-1));
        imagesc(nanmean(I_bout_session{i_behavior},3));
        title(label_meanings{i_behavior})
        set(gca,'xtick',1:10:bout_window*2,'ytick','',...
            'xticklabel',-bout_window:10:bout_window)
        caxis([-2 2]);
    end

    % plot time-locked behavior "activity" around the onset of each behavior
    figure(41); clf
    for i_behavior = 1:size(label_meanings,1)
        subplot(1,size(label_meanings,1),i_behavior); hold on

        % time vectors for plotting
        t1 = -bout_window+1:bout_window;
        t2 = [t1, fliplr(t1)];

        B_bout = B_bout_session{i_behavior};
        % mean
        M = nanmean(B_bout,2);
        % std
        S = nanstd(B_bout,[],2);

        if isempty(M) || isempty(S)
            M = zeros(size(t1,2),1);
            S = zeros(size(t1,2),1);
        end

        % shaded std
        between = [M + S; fliplr(M - S)];

        plot(t1, M, 'k');
        fill(t2, between, [.65 .65 .65], 'edgecolor', 'none', 'facealpha', 0.5)
        title(label_meanings{i_behavior});
        xlabel('time (frames)'); ylabel('Behavior Prob.')
    end
    drawnow
    
end