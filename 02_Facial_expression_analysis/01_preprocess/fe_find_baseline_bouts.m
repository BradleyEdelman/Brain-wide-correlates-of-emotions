function [stim, base, extend] = fe_find_baseline_bouts(face_file, param)
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman


Lbase = param.hog_disp.lbase;
Lstim = param.hog_disp.lstim;
numtrial = param.hog_disp.numtrial;

% load face movement file (extract facial "energy" data)
if exist(face_file,'file'); load(face_file); end

face_diff_tmp = face_diff(1:3500);
[B,I] = sort(face_diff_tmp,'ascend');
I = I(2:end); % first index of a "diff" operation is zero - remove
I(ismember(I, 1:Lbase+1)) = []; % trial cant start before Lbase frames or else not enough baseline
I(ismember(I, 3500 - Lstim + 1:3500)); % trial cant start too close to end or else cant make a full trial

% create baseline "trials"
for i_bout = 1:numtrial
    % start from least motion frame as onset of trial
    bout_start{i_bout} = I(1);
    % create proper bout length
    baseline_bouts{i_bout} = bout_start{i_bout}:bout_start{i_bout} + Lstim - 1;
    % create corresponding baseline
    baseline_bouts_baseline{i_bout} = baseline_bouts{i_bout} - Lbase:baseline_bouts{i_bout} - 1;
    
    % remove all bout and baseline indices from being included in future bouts
    I(ismember(I, [cat(2, baseline_bouts{:}) cat(2, baseline_bouts_baseline{:})])) = [];
end

% find extended trial for each bout
baseline_bouts_extend = cellfun(@(v) v(1) - Lbase:v(1) + 20*Lstim, baseline_bouts, 'uniformoutput', false);

% simple output
stim = baseline_bouts;
base = baseline_bouts_baseline;
extend = baseline_bouts_extend;

idxStim = cat(2, stim{:});
idxBase = cat(2, base{:});
idxExtend = cat(2, extend{:});

% visualize wheel movement in relation to stim periods
figure(11); clf; hold on
time = (1:size(face_diff,2))/20/60;
plot([time(idxBase); time(idxBase)],[min(face_diff) max(face_diff)],'color',[.75 .75 .75]) % baseline
plot([time(idxStim); time(idxStim)],[min(face_diff) max(face_diff)],'color','c') % stimulus
plot(time, face_diff, 'r')

plot([time(3600:3640); time(3600:3640)],[min(face_diff) max(face_diff)],'color','g') % stimulus
