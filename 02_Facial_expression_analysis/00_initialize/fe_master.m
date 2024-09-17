% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman


addpath(genpath(['LOCAL PATH \Facial_Expressions_brad\Facial_expression_code_20220412_Brad'))

% Specify data path details
Data = 'LOCAL PATH \Facial_Expressions_brad\01_sample_data';
Data_HOGS = [Data '_HOGS\'];

% Animal-specific data information
Date = {'20210523', '20210524'};
Mouse = {'mouse_0127'};
idxProto = [1 5 6 6 6]; % quinine, sucrose, tail shock, escape, freeze
idProto = {'quinine', 'sucrose', 'tail_shock', 'escape', 'freeze'};

% analysis parameters
data.data_fold = Data;
data.location = 'local';
data.location = 'network';
data.hog_fold = Data_HOGS;
data.date = Date;
data.mouse = Mouse;
data.idxProto = idxProto;
data.idProto = idProto;

param.pix_per_cell = 24;
param.orient = 8;
param.block_size = 1;
param.reg_spout_crop = 0; % 0: automatic (rough), 1: manual

%%

% check that the emotion prototype folder matches the prototype of interest
fe_check_emotion_prototype_folders(data, param)

% preprocess: crop, register and hog generation
    param.hog_preproc.rewrite = 1;
    param.hog_analyze.dlc_override = 1;
fe_preprocess_hogs(data, param);

% estimate body, wheel and face movement (from general ROI)
    param.movt.rewrite = 0;
fe_video_pixel_movement(data, param);

% create prototypes for specified emotions
    param.proto.rewrite = 0;
    param.hog_disp.numtrial = 7;
    param.hog_disp.lbase = 80;
    param.hog_disp.lstim = 40;
fe_create_prototypes_all(data, param);

% create similarity scores for each dataset and prototype
    param.hog_analyze.rewrite = 0;
fe_HOG_corr_all(data, param)

% normalize similarity scores to examine specificity
    param.hog_analyze.exp = 1;
    param.hog_analyze.exp_val = 5;
fe_proto_normalize_3(data, param)


