function fus_plot_interim_glm_results(glmInfo, stim, c, reg_labels, stim_type, save_file, to_save)

f = figure; clf

tmp = reshape(glmInfo.t_stat,36,1728,size(c,1));
tmp = fus_2d_to_3d(tmp,3);
for i_c = 1:size(c,1)
    subplot(ceil((size(c,1))/4)+1,4,i_c)
    imagesc(tmp(:,:,i_c)); axis off; caxis([-5 5]);
    title(reg_labels{i_c})
end

subplot(ceil((size(c,1))/4)+1,4,ceil((size(c,1))/4)*4+1:ceil((size(c,1))/4)*4+4)
plot(stim + repmat((0:(size(stim,2)-1)), size(stim,1),1))
if size(reg_labels,2) == size(c,1)
    set(gca,'ytick',[0.5:(size(stim,2))],'yticklabel',reg_labels,...
        'fontsize',5,'xtick','')
else
    set(gca,'ytick',[0.5:(size(stim,2))],'yticklabel',['stim' reg_labels],...
        'fontsize',5,'xtick','')
end
title(stim_type)
drawnow


if to_save == 1
    savefig(f, [save_file '.fig'])
    saveas(f, [save_file '.png'])
end

