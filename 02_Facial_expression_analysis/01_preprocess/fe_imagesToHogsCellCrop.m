function [hog, filename] = fe_imagesToHogsCellCrop(imageFolder, param, cropInfo)
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman


pix_per_cell = param.pix_per_cell;
orient = param.orient;
block_size = param.block_size;

files = dir(imageFolder);
filename = {files.name};
filename = filename(cellfun(@(s)~isempty(regexp(s,'.jpg')),filename));

tic
img = imread([imageFolder filename{1}]);
cropCoords = cropInfo.coords;
img = img(cropCoords(1,2):cropCoords(2,2), cropCoords(1,1):cropCoords(2,1));
hog_tmp = extractHOGFeatures(img,...
    'CellSize', [pix_per_cell pix_per_cell], 'NumBins', orient, 'BlockSize', [block_size block_size]);

hog = zeros(size(filename,2), size(hog_tmp,2));
updateWaitbar = waitbarParfor(size(filename,2),'Creating HOGS...');
parfor (i_file = 1:size(filename,2),6)
    updateWaitbar()
    img = imread([imageFolder filename{i_file}]);
    img = img(cropCoords(1,2):cropCoords(2,2), cropCoords(1,1):cropCoords(2,1));
    
    hog(i_file,:) = extractHOGFeatures(img,...
        'CellSize', [pix_per_cell pix_per_cell], 'NumBins', orient, 'BlockSize', [block_size block_size]);
end
toc
