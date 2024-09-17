function [spoutInfo, f, f1] = fe_findSpoutCoords(fold, fold_analysis, param, cropInfo)
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman


if isfolder(fold)
    
    files = dir(fold);
    filename = {files.name};
    filename = filename(cellfun(@(s)~isempty(regexp(s,'-100.jpg')),filename));
    
    img = imread([fold filename{1}]);
    cropCoords = cropInfo.coords;
    img = img(cropCoords(1,2):cropCoords(2,2), cropCoords(1,1):cropCoords(2,1));
    f = figure(1); clf
    imshow(img)
    
    spoutCoords = ginput(2);
    spoutCoords = round(spoutCoords);
    hold on
    rectangle('position',...
        [spoutCoords(1,1), spoutCoords(1,2),...
        spoutCoords(2,1) - spoutCoords(1,1), spoutCoords(2,2) - spoutCoords(1,2)],...
        'linewidth',3,'edgecolor','g')
    
    % again for verification
    f1 = figure(2); clf
    subplot(1,2,1);
    imshow(img); hold on;
    rectangle('position',...
        [spoutCoords(1,1), spoutCoords(1,2),...
        spoutCoords(2,1) - spoutCoords(1,1), spoutCoords(2,2) - spoutCoords(1,2)],...
        'linewidth',3,'edgecolor','g')
    title('labelled spout')
    
    % convert spout coordinates from image dimensions to hog dimensions
    hog_size = floor(size(img)./param.pix_per_cell);
    spoutCoords = floor(spoutCoords./param.pix_per_cell);
    
    % blank spout
    spoutIdx = ones(1,hog_size(1)*hog_size(2)*param.orient);
    
    % find spout indices for one orientation
    tmp_idx = zeros(hog_size);
    tmp_idx(spoutCoords(1,2):spoutCoords(2,2), spoutCoords(1,1):spoutCoords(2,1)) = 1;
    tmp_idx = reshape(tmp_idx,1,[]);
    
    % mask spout indices for all orientations
    for i_orient = 1:param.orient
        spoutIdx(i_orient:param.orient:end) = tmp_idx;
    end
    subplot(1,2,2)
    imagesc(reshape(spoutIdx(1:param.orient:end),cropInfo.size))
    axis equal; axis off
    title('example hog spout removal')
    
    spoutInfo.coords = spoutCoords;
    spoutInfo.idx = spoutIdx;
    spoutInfo.hogSize = cropInfo.size;
    
    savefig(f, [fold_analysis 'spout_boundaries.fig'])
    saveas(f, [fold_analysis 'spout_boundaries.png'])
    
    savefig(f1, [fold_analysis 'spout_hog_verification.fig'])
    saveas(f1, [fold_analysis 'spout_hog_verification.png'])
    
end