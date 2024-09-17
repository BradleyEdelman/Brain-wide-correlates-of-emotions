function clean_hog = fe_hogLoadAndSpoutRemoval(hog_file, param, spoutInfo)
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman


if exist(hog_file,'file')
    while ~exist('TMP','var'); try TMP = load(hog_file); end; end
    load(hog_file); clear TMP
end

clean_hog = hog;
spoutCoords = spoutInfo.coords;
spoutIdx = spoutInfo.idx;
hogSize = spoutInfo.hogSize;
if ~isempty(spoutCoords)
    
    clean_hog(:,spoutIdx == 1) = 0;
    
    % verify spout removal
    hog_tmp = clean_hog(1,:);
    
    f = figure(111); clf
    for i_orient = 1:param.orient
        hog_orient = hog_tmp(i_orient:param.orient:end);
        hog_orient = reshape(hog_orient, hogSize(1), hogSize(2));
        subplot(2, param.orient/2, i_orient);
        imagesc(hog_orient)
    end

end


    

