
close all
clear all

raw_fold.fe = 'F:\Facial_Exp_fUS\Conditioning_videography\';
raw_fold.fus = 'G:\fUS\';
param.loc = 'mpip';

% CORE DATASET
mouse = {'0070', '0071', '0072', '0073', '0122', '0124', '0125', '0127', '0130'}; % CUP and 3Pin

data = fus_data_select(raw_fold, mouse);
data.raw_fold = raw_fold.fus;

param.dt_interp = 0.6;
param.stim.win = [-49 50]; % frames around stimulus for "trial"


%% Pre-processing

    param.load.rewrite = 0; % load and pre-process data
fus_load(data, param) % load data and organize initial data storage/parameters
    
    param.svd.rewrite = 0; % perform SVD decomposition
fus_SVD_interp(data, param)
    
    param.roi.select = 0; % (1) select manual roi, (0) use default roi
    param.roi.rewrite = 0; % extract roi voxel indices
fus_roi(data, param);
    
    param.mask.rewrite = 1; % check mask (overlay on top of PDI "anatomy")
fus_brain_mask(data, param)
    
%% "Artifact Removal"

    param.nobrain.rewrite = 0;
    param.nobrain.thresh = 0.05; % set proportion of components to remove
fus_no_brainer(data, param)

%% GLM TEST - NATIVE SPACE, SINGLE SESSION
    param.video.fs = 20; % sampling rate of videography
    param.glm_emotion.rewrite = 1;
    param.glm_emotion.select = {'quinine', 'sucrose', 'tail_shock'}; % specify emotion prototypes and "other" regressors to include in glm
    param.glm.scrub_stimulus = 1; % scrub stimulus periods from time series
    param.glm.chop_post_stim = 0; % if ~= 0, appends time to stimulus periods
    param.art.suffix = 'nobrainer';
    param.nobrain.thresh = 0.015;
    param.iter.suffix = '_WTA';
    param.note.suffix = '_no_locomotion';

fus_glm_emotion_select_clean(data, param) % individual session GLM
