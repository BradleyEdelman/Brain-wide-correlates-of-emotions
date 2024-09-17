function [stim, base, extend, thresh] = fe_find_freeze_bouts(body_file, face_file, thresh, param)
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman



Lbase = param.hog_disp.lbase;
Lstim = param.hog_disp.lstim;
numtrial = param.hog_disp.numtrial;
[idxStim, idxBase, idxExtend] = fe_idxExtract(3600, 2440, numtrial, Lstim, Lbase);

% load face and body movement file (extract facial "energy" and wheel data)
if exist(body_file,'file') && exist(face_file,'file')

    load(body_file)
    load(face_file)

    % ensure wheel and face time series same length for plotting
    minL = min([size(face_diff,2) size(wheel_diff,2)]);
    face_diff = face_diff(1:minL); wheel_diff = wheel_diff(1:minL);

    % visualize wheel movement in relation to stim periods
    figure(11); clf; hold on
    time = (1:size(wheel_diff,2))/20/60;

        idxBase(idxBase > size(time,2)) = [];
        idxStim(idxStim > size(time,2)) = [];

    plot([time(idxBase); time(idxBase)],[-max(face_diff) max(wheel_diff)],'color',[.75 .75 .75]) % baseline
    plot([time(idxStim); time(idxStim)],[-max(face_diff) max(wheel_diff)],'color','c') % stimulus
    plot(time, wheel_diff, 'k')
    plot(time, -face_diff, 'r')

    % threshold for high movement based on individual run
    if isempty(thresh.movt)
        thresh.movt = 0;
        while thresh.movt == 0 || ~isnumeric(thresh.movt)
            thresh.movt = input('\nenter threshold for low movement\n');
        end
    end

    % threshold for no orofacial movt based on individual run
    if isempty(thresh.face)
        thresh.face = 0;
        while thresh.face == 0 || ~isnumeric(thresh.face)
            thresh.face = input('\nenter threshold for no orofacial movt\n');
        end
    end

    low_movt = find(wheel_diff < thresh.movt & face_diff < thresh.face);

    % find consecutive frames above threshold
    freeze_bouts = mat2cell(low_movt, 1, diff([0, find(diff(low_movt) ~= 1), length(low_movt)] ));
    % apply bout length threshold
    freeze_bouts(cellfun('length', freeze_bouts) < thresh.freeze) = [];
    freeze_bouts_plot = freeze_bouts;

    % find bout onset and offset
    bout_onset = cellfun(@(v)v(1), freeze_bouts);
    bout_offset = cellfun(@(v)v(end), freeze_bouts);

    % ensure each bout has a bout-less baseline
    for i_bout = size(bout_onset,2):-1:2 % start from 2 since we take care of first one later
        if bout_onset(i_bout) - Lbase < bout_offset(i_bout-1) % make sure previous bout ends before baseline for new bout begins
            freeze_bouts(i_bout) = [];
            bout_onset(i_bout) = [];
            bout_offset(i_bout) = []; 
        end
    end

    % ensure first bout has sufficient baseline
    freeze_bouts(bout_onset < Lbase) = [];
    bout_offset(bout_offset < Lbase) = [];
    bout_onset(bout_onset < Lbase) = [];

    % ensure last bout has sufficient end baseline
    freeze_bouts(bout_onset + 4*Lbase > size(wheel_diff,2)) = [];
    bout_offset(bout_offset + 4*Lbase > size(wheel_diff,2)) = [];
    bout_onset(bout_onset + 4*Lbase > size(wheel_diff,2)) = [];

    % only use bouts in fearful context (after first shock) and not during tailshock baseline/stim
    freeze_bouts(cellfun(@(v) sum(ismember(1:3600, v)) ~= 0, freeze_bouts)) = [];
    freeze_bouts(cellfun(@(v) sum(ismember(idxStim, v)) ~= 0, freeze_bouts)) = [];
    freeze_bouts(cellfun(@(v) sum(ismember(idxBase, v)) ~= 0, freeze_bouts)) = [];

    % find baseline for each "clean" bout
    freeze_bouts_baseline = cellfun(@(v) v(1) - Lbase:v(1) - 1, freeze_bouts, 'uniformoutput', false);
    % find extended trial for each bout
    freeze_bouts_extend = cellfun(@(v) v(1) - Lbase:v(1) + 20*Lstim, freeze_bouts, 'uniformoutput', false);

    % simple output
    stim = freeze_bouts;
    base = freeze_bouts_baseline;
    extend = freeze_bouts_extend;

else

    stim = cell(0);
    base = cell(0);
    extend =cell(0);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualization
%{

time = (1:size(wheel_diff,2))/20/60;
freeze_bouts_plot = cat(2, freeze_bouts_plot{:});
freeze_ts = nan(1,size(time,2));
freeze_ts(freeze_bouts_plot) = 2;

% plot all freeze bouts
figure(14); clf; hold on
wheel_diff = rescale(wheel_diff);
face_diff = rescale(face_diff);
plot([time(idxBase); time(idxBase)],[0 2.1],'color',[.75 .75 .75]) % baseline
plot([time(idxStim); time(idxStim)],[0 2.1],'color','c') % stimulus
plot(time, freeze_ts,'color',[31 55 104]/255,'linewidth',10) % stimulus
yyaxis left; plot(time, movmean(wheel_diff,10), 'k','linewidth',.5)
set(gca,'ylim',[0 2.1],'ytick',''); ylabel('norm. movt. (wheel)');
yyaxis right; plot(time, movmean(face_diff,10) + 1, 'r','linewidth',.5)
set(gca,'ylim',[0 2.1],'ytick','','YColor','r'); ylabel('norm. movt. (face)');
xlabel('time (min)')


% plot "selected" escape trials
figure(15); clf; hold on
IDX = randperm(size(stim,2)); IDX = IDX(1:numtrial);
stim = cellfun(@(v)v(1:Lstim), stim, 'uniformoutput', false);
plot([time(idxBase); time(idxBase)],[0 2.1],'color',[.75 .75 .75]) % baseline
plot([time(idxStim); time(idxStim)],[0 2.1],'color','c') % stimulus
base_ts = nan(1,size(time,2)); base_ts(cat(2, base{IDX})) = 2;
stim_ts = nan(1,size(time,2)); stim_ts(cat(2, stim{IDX})) = 2;
plot(time, stim_ts,'color',[31 55 104]/255,'linewidth',10) % stimulus
plot(time, base_ts,'color',[.75 .75 .75],'linewidth',10) % stimulus
yyaxis left; plot(time, movmean(wheel_diff,10), 'k','linewidth',.5)
set(gca,'ylim',[0 2.1],'ytick',''); ylabel('norm. movt. (wheel)');
yyaxis right; plot(time, movmean(face_diff,10) + 1, 'r','linewidth',.5)
set(gca,'ylim',[0 2.1],'ytick','','YColor','r'); ylabel('norm. movt. (face)');
xlabel('time (min)')

%}

