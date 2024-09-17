function [cropInfo, f, f1] = fe_findCropCoords(fold, fold_analysis, param)
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman


if isfolder(fold)
    
    files = dir(fold);
    filename = {files.name};
    filename = filename(cellfun(@(s)~isempty(regexp(s,'-100.jpg')),filename));
    img = imread([fold filename{1}]);
    
elseif exist(fold{1},'file')
    
    filename = fold;
    img = imread(filename{1});
    
end
    

f = figure(1); clf
imshow(img)

cropCoords = ginput(2);
cropCoords = round(cropCoords);
cropCoords(1,1) = 1; cropCoords(2,1) = size(img,2);
hold on
rectangle('position',...
    [cropCoords(1,1), cropCoords(1,2),...
    cropCoords(2,1) - cropCoords(1,1), cropCoords(2,2) - cropCoords(1,2)],...
    'linewidth',3,'edgecolor','g')
title('original image, crop boundaries')

img_crop = img(cropCoords(1,2):cropCoords(2,2), cropCoords(1,1):cropCoords(2,1));
f1 = figure(2); clf
subplot(1,2,1); imshow(img)
title('original image')
subplot(1,2,2); imshow(img_crop)
title('cropped image')

% convert spout coordinates from image dimensions to hog dimensions
cropSize = floor(size(img_crop)./param.pix_per_cell);

cropInfo.coords = cropCoords;
cropInfo.size = cropSize;

savefig(f, [fold_analysis 'crop_boundaries.fig'])
saveas(f, [fold_analysis 'crop_boundaries.png'])

savefig(f1, [fold_analysis 'cropped_image.fig'])
saveas(f1, [fold_analysis 'cropped_image.png'])
    
    
