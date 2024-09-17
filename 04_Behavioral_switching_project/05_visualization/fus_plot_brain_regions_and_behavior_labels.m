function fus_plot_brain_regions_and_behavior_labels(new_name, I_seg, labels, param)

if param.plot_session == 1

    figure(101); clf;
    subplot(20,20,1:20:380);
    % find and plot "big" region classes
    big_reg = unique(new_name(:,2));
    for i_big_reg = 1:size(big_reg,1)
        big_regC(strcmp(new_name(:,2), big_reg(i_big_reg))) = i_big_reg;
    end
    cmap = flipud(colorcube(size(big_regC,2)));
    imagesc(big_regC'); set(gca,'colormap', cmap); axis off
    ylabel('"Big" region breakdown')

    % plot data from all regions over time
    big_plot = 1:400; big_plot = reshape(big_plot,20,20); big_plot(1,:) = []; big_plot(:,end) = [];
    subplot(20,20,big_plot(:)');
    imagesc(I_seg);
    axis off; caxis([-2 2])

    % plot behavior labels over time
    subplot(20,20,382:400)
    labels = labels(:)';
    imagesc(labels);
    cmap1 = [0 0 0; 1 0 0; 0 1 0; 0 0 1; 1 1 0];
    set(gca,'colormap',cmap1); axis off
    
end