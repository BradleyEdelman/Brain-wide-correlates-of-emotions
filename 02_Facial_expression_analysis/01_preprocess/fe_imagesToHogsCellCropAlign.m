function hog = fe_imagesToHogsCellCropAlign(imageFolder, referenceFolder, param, cropInfo, varargin)
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman


pix_per_cell = param.pix_per_cell;
orient = param.orient;
block_size = param.block_size;

reg_spout_crop = param.reg_spout_crop;

%% Registration

% check if filenames included already from previous processes
tmp = strcmp('imageFiles',varargin);
if sum(tmp) > 0
    
    idx = find(tmp == 1);
    filename = varargin{idx + 1};
    
else
    
    files = dir(imageFolder);
    filename = {files.name};
    filename = filename(cellfun(@(s)~isempty(regexp(s,'.jpg')),filename));
    
end

tmp = strcmp('referenceFiles',varargin);
if sum(tmp) > 0
    
    idx = find(tmp == 1);
    filenameR = varargin{idx + 1};
    
else
    
    filesR = dir(referenceFolder);
    filenameR = {filesR.name};
    filenameR = filenameR(cellfun(@(s)~isempty(regexp(s,'.jpg')),filenameR));
    
end

tmp = strcmp('saveFolder',varargin);
if sum(tmp) > 0
    idx = find(tmp == 1);
    saveFolder = varargin{idx + 1};
end

%% show misaligned images

% load DLC files for each dataset to determine "clean" frames for registration
rparts = strsplit(referenceFolder,'\');
referenceHOGSfold = [rparts{1} '\' rparts{2} '\' rparts{3} '_HOGS\' rparts{4}...
    '\' rparts{5} '\' rparts{6} '\' rparts{7} '\'];
referenceDLC = [referenceHOGSfold 'invalid_indices_2paw.mat'];
if exist(referenceDLC,'file')
    load(referenceDLC);
    idxRgood = find(c == 1);
else
    idxRgood = ones(1,size(filenameR,2));
end

iparts = strsplit(imageFolder,'\');
imageHOGSfold = [iparts{1} '\' iparts{2} '\' iparts{3} '_HOGS\' iparts{4}...
    '\' iparts{5} '\' iparts{6} '\' iparts{7} '\'];
imageDLC = [imageHOGSfold 'invalid_indices_2paw.mat'];
if exist(imageDLC,'file')
    load(imageDLC)
    idxIgood = find(c == 1);
else
    idxIgood = ones(1,size(filename,2));
end


% Not worth the time to extract hogs if no video analysis to ensure clean
% data (unless overridden)
if exist(referenceDLC,'file') && exist(imageDLC,'file') || param.hog_analyze.dlc_override == 1

    %%%%%%%% when picking neutral frames, look at motion energy to identify
    %%%%%%%% frames with little changing (indicates resting state)

    L = min([100 min([size(idxRgood,1) size(idxIgood,1)])]);
    % take L random clean images from each image stack for registration
    ridxRgood = idxRgood(randperm(size(idxRgood,1))); 
    ridxIgood = idxIgood(randperm(size(idxIgood,1)));
    for i_iter = 1:L
%         i_iter
        fixed(:,:,i_iter) = imread([referenceFolder filenameR{ridxRgood(i_iter)}]);
        moving(:,:,i_iter) = imread([imageFolder filename{ridxIgood(i_iter)}]);
    end
    fixed = mean(fixed,3); moving = mean(moving,3);


    %% crop spout from both images to avoid misregistration

    if reg_spout_crop == 1
        f = figure(20); clf
        subplot(2,2,1); imagesc(fixed);
        title('fixed: original'); axis equal, axis off
        spoutCoords1 = ginput(2);
        spoutCoords1 = round(spoutCoords1);
        fixed(spoutCoords1(1,2):spoutCoords1(2,2), spoutCoords1(1,1):spoutCoords1(2,1)) = 0;
        subplot(2,2,2); imagesc(fixed);
        title('fixed: manual spout crop'); axis equal; axis off

        subplot(2,2,3); imagesc(moving)
        title('moving: original'); axis equal, axis off
        spoutCoords2 = ginput(2);
        spoutCoords2 = round(spoutCoords2);
        moving(spoutCoords2(1,2):spoutCoords2(2,2), spoutCoords2(1,1):spoutCoords2(2,1)) = 0;
        subplot(2,2,4); imagesc(moving);
        title('moving: manual spout crop'); axis equal; axis off

    else

        f = figure(20); clf
        subplot(2,2,1); imagesc(fixed);
        title('fixed: original'); axis equal, axis off
        fixed(700:end, 1100:end) = 0;
        subplot(2,2,2); imagesc(fixed);
        title('fixed: rough spout crop'); axis equal; axis off

        subplot(2,2,3); imagesc(moving);
        title('moving: original'); axis equal, axis off
        moving(700:end, 1100:end) = 0;
        subplot(2,2,4); imagesc(moving);
        title('moving: rough spout crop'); axis equal; axis off

    end
    %%

    % register images
    f1 = figure(1); clf
    subplot(1,2,1)
    imshowpair(fixed, moving, 'scaling', 'joint');
    title('pre-registration')
    drawnow

    % estimate transformation
    [optimizer, metric] = imregconfig('monomodal');
    tform = imregtform(moving, fixed, 'rigid', optimizer, metric);
    
    % ensure no crazy weird transformations; if so, try again max 10x
    if abs(tform.T(1,2)) > 0.1 || abs(tform.T(2,1)) > 0.1 
        tform = imregtform(moving, fixed, 'translation', optimizer, metric);
    end
    
    % apply/check transformation
    movingReg = imwarp(moving,tform,'OutputView',imref2d(size(fixed)));
    subplot(1,2,2); cla
    imshowpair(fixed, movingReg, 'scaling', 'joint');
    title('post-registration')
    drawnow
    

    %% save figures if desired
    if exist('saveFolder','var')
        savefig(f, [saveFolder 'hog_registration_spout_crop.fig'])
        saveas(f, [saveFolder 'hog_registration_spout_crop.png'])

        savefig(f1, [saveFolder 'hog_registration.fig'])
        saveas(f1, [saveFolder 'hog_registration.png'])
    end

    %% register and exgtract hogs

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

        % register
        imgReg = imwarp(img,tform,'OutputView',imref2d(size(fixed)));

        imgRegCrop = imgReg(cropCoords(1,2):cropCoords(2,2), cropCoords(1,1):cropCoords(2,1));
        hog(i_file,:) = extractHOGFeatures(imgRegCrop,...
            'CellSize', [pix_per_cell pix_per_cell], 'NumBins', orient, 'BlockSize', [block_size block_size]);
    end
    toc
else
    
    hog = [];

end
