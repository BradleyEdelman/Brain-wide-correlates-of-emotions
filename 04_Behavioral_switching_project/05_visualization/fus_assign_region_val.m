function region_img = fus_assign_region_val(values, idx, new_seg, plot, fig_num)

if size(values,2) > 1
    values = nanmean(values,2);
end

if isempty(idx)
    idx = 1:size(values,1);
end


region_img = nan(size(new_seg));
cnt = 1;
for i_reg = min(new_seg(:)):max(new_seg(:))
    if ismember(i_reg,idx)
        region_img(new_seg == i_reg) = values(cnt);
        cnt = cnt + 1;
    else
        region_img(new_seg == i_reg) = 0;
    end
end

% region_img = region_img(:,:,7:3:end-7);
region_img = reshape(region_img,size(region_img,1),[]);
region_img = fus_2d_to_3d(region_img,7);

if plot == 1
    if isempty(fig_num)
        figure(randi(100,1)); clf
    else
        figure(fig_num); clf
    end
    imagesc(region_img)
    caxis([-1 1]);
    colormap([.85 .85 .85; fireice])
end


    
    

