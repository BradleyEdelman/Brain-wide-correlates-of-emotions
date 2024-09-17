function labels = fus_extract_behavior_labels(I, label_files, param)
% Mace Lab - Max Planck Institute of Neurobiology
% Author: Bradley Edeman
% Date: 10.08.22

%%
if strcmp(param.behavior.labels,'original')

    % load behavior labels
    labels = load(label_files{1}); % load behavior labels
    labels = labels(:); % force tall matrix

    % ADJUST LABELS ACCORDING TO PATRICK NOTES
    labels = [zeros(100,1); labels]; % add 100 blank frames to the beginning
    labels(end-99:end) = 0; % set last 100 indices to zero (patrick)

    % bin behavior labels into fus sampling rate
    fs_fus = 1/param.fus.dt; % sampling rate of fus
    fs_behav = 30; % sampling rate of video
    oversampling = ceil(fs_behav/fs_fus); % number of video frames that fit into a single ultrasound frame

    % ensure beahvior label time series has sufficient number of frames for binning
    REM = rem(size(labels,1), oversampling);
    % append or remove extra nan values to ensure same number of bins as fus frames
    if size(labels,1)/oversampling > size(I,2) 
        labels = labels(1:end-REM); % if too many frames, remove extras
    elseif size(labels,1)/oversampling < size(I,2)
        labels = [labels; nan(oversampling - REM,1)]; % if too few frames, add difference to make a full bin
    end

    % reshape behavior labels into bins
    labels = reshape(labels, oversampling, []);
    % find most common (mode) behavior during each bin
    labels = mode(labels,1);
    % interpolate across any frames that are nan
    labels = fillmissing(labels,'linear',2); 
    labels = labels(:); % force tall matrix   
    
elseif strcmp(param.behavior.labels,'corrected')
    
    % load behavior labels
    load(label_files{2}); % load corrected behavior labels
    labels = corrected_labels(:); % force tall matrix
    
elseif strcmp(param.behavior.labels,'whisking')
    
    load(label_files{3}); % load corrected behavior labels
    labels = labels_1234'; % force tall matrix
    
end