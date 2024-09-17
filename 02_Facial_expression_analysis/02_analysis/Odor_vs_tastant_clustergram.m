% Check clustergrams of aversive/appetetive odors/tastants
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman



% Specify data path details
Data = '\\nas6\datastore_brad$\Facial_Exp_fUS';
Data_HOGS = [Data '_HOGS\'];

data.data_fold = Data;
data.location = 'local';
data.location = 'network';
data.hog_fold = Data_HOGS;
data.date = Date;
data.mouse = Mouse;

param.pix_per_cell = 24;
param.orient = 8;
param.block_size = 1;

Mouse = {'mouse_1002', 'mouse_1003', 'mouse_1006', 'mouse_1007', 'mouse_1008'};
Date = {'20211101', '20211102', '20211103', '20211110', '20211115', '20211116', '20211117', '20211118'};
%%

pair{1} = {'sucrose','2PE'}; cpair{1} = [0 .7 .3; .5 1 .5; .6 .6 .6; .9 .9 .9];
pair{2} = {'quinine','isoP'}; cpair{2} = [.5 .2 .6; .7 .5 .8; .6 .6 .6; .9 .9 .9];

[idxStim, idxBase, idxExtend] = fe_idxExtract(3600, 2440, 5, 40, 80);
for i_mouse = 1:size(Mouse,2)
    
    % specify image folders and corresponding hog folders to save to
    [fold_face, fold_face_hog] = fe_identify_video_fold(data, i_mouse, 'FACE');
    
    % specify crop and spout coord file from preprocessing
    coord_file = [fold_face_hog{1} 'crop_coord.mat'];
    if exist(coord_file,'file')
        load(coord_file)
    end
    
    spout_coord_file = [fold_face_hog{1} 'spout_coord.mat'];
    if exist(spout_coord_file,'file')
        load(spout_coord_file)
    end
    
    for i_pair = 1:size(pair,2)
    
        % find STIMULUS runs for reference
        solicit_idx = cell(0);
        for i_stim = 1:size(pair{i_pair},2) % skip neutral
            stim_idx = find(contains(fold_face,pair{i_pair}{i_stim}));
            solicit_idx{i_stim} = stim_idx;
        end
        
        for i_stim = 1:size(solicit_idx,2)
            
            hog_file = [fold_face_hog{solicit_idx{i_stim}} 'hogs.mat'];
            if exist(hog_file,'file')
                
                % load hogs and remove spout indices
                hog = fe_hogLoadAndSpoutRemoval(hog_file, param, spoutInfo);
                
                hog_base{i_stim} = hog(idxBase,:);
                hog_stim{i_stim} = hog(idxStim,:);
            
            end
            
            cluster_hog = [hog_stim{i_stim}; hog_base{i_stim}];
            labels = [ones(1,size(hog_stim{i_stim},1)) zeros(1,size(hog_base{i_stim},1))];
            [fc, leafOrder] = fe_clustergram(cluster_hog, labels);
        end
        
        cluster_hog = [hog_stim{1}; hog_stim{2}; hog_base{1}; hog_base{2}];
        labels = [ones(1,size(hog_stim{1},1)) 2*ones(1,size(hog_stim{2},1))...
            3*ones(1,size(hog_base{1},1)) 4*ones(1,size(hog_base{2},1))];
        [fc, leafOrder] = fe_clustergram(cluster_hog, labels);
        set(gca,'colormap',cpair{i_pair})
        title(['mouse #: ' num2str(i_mouse)])
        pause
    
    end
end