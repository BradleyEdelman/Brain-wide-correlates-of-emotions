function [new_seg, new_name] = fus_custom_segment_PW(file, atlas, plot)
% Mace and Gogolla Labs - Max Planck Institute of Neurobiology
% Author: Bradley Edeman
% Date: 03.11.21

%% segmentation of allen atlas according to regions defined in corresponding excel file
% file - excel file detailing which regions from the allen atlas to include and combine together
% atlas - allen atlas
% plot - flag for plotting segmentation regions (1 - plot, 0 - no plot)

%%

if isempty(file)
    file = 'R:\Work - Experiments\Codes\Paulina\functions\Consolidated_regions_brad.xlsx';
end

%%
seg = atlas.Regions_crop;
names = atlas.Names';

[NUM,TXT,RAW]=xlsread(file);

new_seg = nan(size(seg));
new_name = cell(0); grp_name = cell(0);
cnt = 1;
for i_reg = 2:size(TXT,1)
    new_reg = TXT(i_reg,:);
    
    % first line of areas contains major category (cortex, striatum, etc.)
    if ~isempty(new_reg{1})
        
        grp_name = new_reg{1};
        
    else
    
        if isempty(new_reg{2}) % not an aggregate region

            % find region name in atlas
            idx = find(strcmp(new_reg{3}, names));
            new_name{cnt,1} = new_reg{3};
            new_seg(seg == idx) = cnt;

        else % new region comprised of multiple atlas regions (need to combine)

            num_reg = size(find(~cellfun(@isempty,new_reg)),2);
            for ii_reg = 1:num_reg-1
                idx = find(strcmp(new_reg{2+ii_reg}, names));
                new_seg(seg == idx) = cnt;
            end
            new_name{cnt,1} = new_reg{2};

        end
        new_name{cnt,2} = grp_name;
        
        cnt = cnt +1;
        
    end
    
end
%%
lateral = 0; % 1 yes, 0 no
if lateral == 1
%% implement laterality
    % find regions that are continuous across midline
    midline = size(new_seg,2)/2;
    central = new_seg(:,midline - 1:midline + 1,:); % look at two voxel l/r of midline
    central = reshape(permute(central,[2 3 1]),size(central,2),[])'; % stack all midlines
    central(sum(isnan(central),2)== 3,:) = []; % remove nan rows
    central(isnan(central(:,2)),:) = []; % remove rows with central nan value

    central_num = [];
    for i_row = 1:size(central,1) 
        tmp_row = central(i_row,:); % look at each row
        tmp_row(isnan(tmp_row)) = []; % remove nans
        tmp_unique = unique(tmp_row); % find unique values
        if size(tmp_unique,2) == 1
            central_num = [central_num tmp_unique]; % keep if only one unique value
        end
    end
    [GC, GR] = groupcounts(central_num'); % find how many rows each unqiue region crosses midline
    GR(GC < 35) = []; % threshold number of times region crosses midline to be counted

    central_reg = new_seg;
    central_reg(~ismember(new_seg, GR)) = 0;
    central_reg(isnan(new_seg)) = nan;

    % viz for verification
    tmp = central_reg(1:2:end,1:2:end,1:2:end);
    tmp = tmp(:,:,param.slice_tot);
    tmp = tmp(:,:,param.slice_viz);
    tmp = reshape(tmp,size(tmp,1),[]); tmp = fus_2d_to_3d(tmp,3);
    figure(101); clf;
    subplot(2,2,1); imagesc(tmp);
    caxis([-1 max(tmp(:))]); cmap = [.65 .65 .65; .85 .85 .85; jet(max(tmp(:)))];
    set(gca,'colormap',cmap); axis off

    % adjust seg and name list to accomodate left/right/central regions
    num_reg_orig = size(new_name,1);
    for i_reg = 1:num_reg_orig
        if ~ismember(i_reg, GR) % if not a central region
            new_idx = size(new_name,1) + 1;
            tmp_name = new_name{i_reg,1};
            tmp_lobe = new_name{i_reg,2};

            new_name{i_reg,1} = ['L ' tmp_name]; % set existing name index to left
            new_name{new_idx,1} = ['R ' tmp_name]; % append right name index to end
            new_name{new_idx,2} = tmp_lobe;

            tmp_seg = new_seg;
            tmp_seg(tmp_seg ~= i_reg) = 0; % find all brain indices belonging to region
            tmp_seg(:,1:midline,:) = 0; % ignore all left side region indices
            new_seg(tmp_seg == i_reg) = new_idx; % set right side indices to new region index

        end
    end

    % find indices of left, right and central regions
    regL = find(contains(new_name(:,1),'L '));
    regR = find(contains(new_name(:,1),'R '));
    regC = find(~contains(new_name(:,1),'L ') & ~contains(new_name(:,1),'R '));
    
    % vizualize lateral regions again for confirmation
    left_seg = new_seg; % left regions
    left_seg(~ismember(left_seg,regL)) = 0;
    left_seg(isnan(new_seg)) = nan;
    tmp = left_seg(1:2:end,1:2:end,1:2:end);
    tmp = tmp(:,:,param.slice_tot);
    tmp = tmp(:,:,param.slice_viz);
    tmp = reshape(tmp,size(tmp,1),[]); tmp = fus_2d_to_3d(tmp,3);
    subplot(2,2,2); imagesc(tmp);
    caxis([-1 max(tmp(:))]); cmap = [.65 .65 .65; .85 .85 .85; jet(max(tmp(:)))];
    set(gca,'colormap',cmap); axis off
    
    right_seg = new_seg; % right regions
    right_seg(~ismember(right_seg,regR)) = 0;
    right_seg(isnan(new_seg)) = nan;
    tmp = right_seg(1:2:end,1:2:end,1:2:end);
    tmp = tmp(:,:,param.slice_tot);
    tmp = tmp(:,:,param.slice_viz);
    tmp = reshape(tmp,size(tmp,1),[]); tmp = fus_2d_to_3d(tmp,3);
    subplot(2,2,3); imagesc(tmp);
    caxis([-1 max(tmp(:))]); cmap = [.65 .65 .65; .85 .85 .85; jet(max(tmp(:)))];
    set(gca,'colormap',cmap); axis off
    
    lat_seg = new_seg;
    lat_seg(ismember(lat_seg, regL)) = 1;
    lat_seg(ismember(lat_seg, regR)) = 2;
    lat_seg(ismember(lat_seg, regC)) = 3;

    tmp = lat_seg(1:2:end,1:2:end,1:2:end);
    tmp = tmp(:,:,param.slice_tot);
    tmp = tmp(:,:,param.slice_viz);
    tmp = reshape(tmp,size(tmp,1),[]); tmp = fus_2d_to_3d(tmp,3);
    subplot(2,2,4); imagesc(tmp);
    caxis([-1 3.5]); cmap = [.65 .65 .65; 1 0 0; 0 1 0; 0 0 1];
    set(gca,'colormap',cmap); axis off

    new_seg = new_seg(1:2:end,1:2:end,1:2:end);
    new_seg = new_seg(:,:,param.slice_tot);
    
    lobe_name = unique(new_name(:,2));
    lobe_segL = zeros(size(new_seg));
    lobe_segR = zeros(size(new_seg));
    lobe_segC = zeros(size(new_seg));
    for i_lobe = 1:size(lobe_name,1)
        lobe_idx = find(strcmp(new_name(:,2),lobe_name(i_lobe)));

        % identify lobe for left, right and central regions
        lobe_idxL = intersect(lobe_idx,regL);
        lobe_segL(ismember(new_seg,lobe_idxL)) = i_lobe;

        lobe_idxR = intersect(lobe_idx,regR);
        lobe_segR(ismember(new_seg,lobe_idxR)) = i_lobe;

        lobe_idxC = intersect(lobe_idx,regC);
        lobe_segC(ismember(new_seg,lobe_idxC)) = i_lobe;
    end

    % zero pad on L and R to accomodate visual separation
    [nx nz ny] = size(new_seg);
    lobe_segL_viz = [zeros(nx, 30, ny) lobe_segL zeros(nx, 30, ny)];
    lobe_segR_viz = [zeros(nx, 30, ny) lobe_segR zeros(nx, 30, ny)];
    lobe_segC_viz = [zeros(nx, 30, ny) lobe_segC zeros(nx, 30, ny)];

    % shift left and right regions away from central regions for vis
    lobe_segL_viz = circshift(lobe_segL_viz, -20, 2);
    lobe_segR_viz = circshift(lobe_segR_viz, 20, 2);

    % add left, right and central seg together
    lobe_seg_viztot = lobe_segL_viz + lobe_segR_viz + lobe_segC_viz;
    lobe_seg_viztot(lobe_seg_viztot == 0) = nan;

    % vis
    tmp = lobe_seg_viztot; tmp = tmp(:,:,param.slice_viz);
    tmp = reshape(tmp,80,[]); tmp = fus_2d_to_3d(tmp,3);
    figure(103); clf;
    imagesc(tmp);
    cmap1 = flipud(colorcube(size(lobe_name,1)));
    caxis([0 size(lobe_name,1)]); cmap = [.65 .65 .65; cmap1]; set(gca,'colormap',cmap);

else
	 
    % find different groupings of regions
    grp = [0; find(isnan(NUM)); 500]; grp(3:end) = grp(3:end) - (1:size(grp,1)-2)';
    new_seg_grp = nan(size(new_seg));
    for i_grp = 2:size(grp,1)
        new_seg_grp(new_seg >= grp(i_grp - 1) & new_seg < grp(i_grp)) = i_grp-1;
    end
    
    if plot == 1
    
        tmp = new_seg; tmp = tmp(:,:,7:3:end-7); % look at only a few slices for verification
        tmp = reshape(tmp,size(tmp,1),[]); tmp = fus_2d_to_3d(tmp,3);
        figure(100); clf;
        subplot(1,2,1); imagesc(tmp);
        caxis([-10 max(tmp(:))]); cmap = [.65 .65 .65; jet(128)]; colormap(cmap)
        axis off

        tmp = new_seg_grp; tmp = tmp(:,:,7:3:end-7);
        tmp = reshape(tmp,size(tmp,1),[]); tmp = fus_2d_to_3d(tmp,3);
        subplot(1,2,2); imagesc(tmp);
        cmap1 = flipud(colorcube(size(grp,1)-1));
        caxis([0 size(grp,1)]); cmap = [.65 .65 .65; cmap1]; set(gca,'colormap',cmap);
        axis off
        
    end
    
end


