function [stim, base, extend, thresh] = fe_find_escape_bouts(body_file, thresh, param)
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman



Lbase = param.hog_disp.lbase;
Lstim = param.hog_disp.lstim;
numtrial = param.hog_disp.numtrial;
[idxStim, idxBase, idxExtend] = fe_idxExtract(3600, 2440, numtrial, Lstim, Lbase);

% load body movement file (extract wheel data)
if exist(body_file,'file')
    
    load(body_file)
    
    % visualize wheel movement in relation to stim periods
    figure(11); clf; hold on
    time = (1:size(wheel_diff,2))/20/60;

        idxBase(idxBase > size(time,2)) = [];
        idxStim(idxStim > size(time,2)) = [];

    plot([time(idxBase); time(idxBase)],[0 max(wheel_diff)],'color',[.75 .75 .75]) % baseline
    plot([time(idxStim); time(idxStim)],[0 max(wheel_diff)],'color','c') % stimulus
    plot(time, wheel_diff, 'k')

    % threshold for high movement based on individual run
    if isempty(thresh.movt)
        thresh.movt = 0;
        while thresh.movt == 0 || ~isnumeric(thresh.movt)
            thresh.movt = input('\nenter threshold for high movement\n');
        end
    end

    high_movt = find(wheel_diff > thresh.movt);

    % find consecutive frames above threshold
    escape_bouts = mat2cell(high_movt, 1, diff([0, find(diff(high_movt) ~= 1), length(high_movt)] ));
    % apply bout length threshold
    escape_bouts(cellfun('length', escape_bouts) < thresh.escape) = [];
    escape_bouts_plot = escape_bouts;

    % find bout onset and offset
    bout_onset = cellfun(@(v)v(1), escape_bouts);
    bout_offset = cellfun(@(v)v(end), escape_bouts);

    % ensure each bout has a bout-less baseline
    for i_bout = size(bout_onset,2):-1:2 % start from 2 since we take care of first one later
        if bout_onset(i_bout) - Lbase < bout_offset(i_bout-1) % make sure previous bout ends before baseline for new bout begins
            escape_bouts(i_bout) = [];
            bout_onset(i_bout) = [];
            bout_offset(i_bout) = []; 
        end
    end

    % ensure first bout has sufficient baseline
    escape_bouts(bout_onset < Lbase) = [];
    bout_offset(bout_offset < Lbase) = [];
    bout_onset(bout_onset < Lbase) = [];

    % ensure last bout has sufficient end baseline
    escape_bouts(bout_onset + 4*Lbase > size(wheel_diff,2)) = [];
    bout_offset(bout_offset + 4*Lbase > size(wheel_diff,2)) = [];
    bout_onset(bout_onset + 4*Lbase > size(wheel_diff,2)) = [];

    % only use bouts in fearful context (after first shock) and not during tailshock baseline/stim
    escape_bouts(cellfun(@(v) sum(ismember(1:3600, v)) ~= 0, escape_bouts)) = [];
    escape_bouts(cellfun(@(v) sum(ismember(idxStim, v)) ~= 0, escape_bouts)) = [];
    escape_bouts(cellfun(@(v) sum(ismember(idxBase, v)) ~= 0, escape_bouts)) = [];

    % find baseline for each "clean" bout
    escape_bouts_baseline = cellfun(@(v) v(1) - Lbase:v(1) - 1, escape_bouts, 'uniformoutput', false);
    % find extended trial for each bout
    escape_bouts_extend = cellfun(@(v) v(1) - Lbase:v(1) + 20*Lstim, escape_bouts, 'uniformoutput', false);

    % simple output
    stim = escape_bouts;
    base = escape_bouts_baseline;
    extend = escape_bouts_extend;

else

    stim = cell(0);
    base = cell(0);
    extend = cell(0);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualization
%{

time = (1:size(wheel_diff,2))/20/60;
escape_bouts_plot = cat(2, escape_bouts_plot{:});
escape_ts = nan(1,size(time,2));
escape_ts(escape_bouts_plot) = 1;

% plot all escape bouts
figure(14); clf; hold on
wheel_diff = rescale(wheel_diff);
plot([time(idxBase); time(idxBase)],[0 1.1],'color',[.75 .75 .75]) % baseline
plot([time(idxStim); time(idxStim)],[0 1.1],'color','c') % stimulus
plot(time, escape_ts,'color',[63 110 184]/255,'linewidth',10) % stimulus
plot(time, movmean(wheel_diff,10), 'k','linewidth',.5)
set(gca,'ylim',[0 1.1],'ytick','');
ylabel('norm. movt. (wheel)') ; xlabel('time (min)')


% plot "selected" escape trials
figure(15); clf; hold on
IDX = randperm(size(stim,2)); IDX = IDX(1:numtrial);
stim = cellfun(@(v)v(1:Lstim), stim, 'uniformoutput', false);
plot([time(idxBase); time(idxBase)],[0 1.1],'color',[.75 .75 .75]) % baseline
plot([time(idxStim); time(idxStim)],[0 1.1],'color','c') % stimulus
base_ts = nan(1,size(time,2)); base_ts(cat(2, base{IDX})) = 1;
stim_ts = nan(1,size(time,2)); stim_ts(cat(2, stim{IDX})) = 1;
plot(time, stim_ts,'color',[63 110 184]/255,'linewidth',10) % stimulus
plot(time, base_ts,'color',[.75 .75 .75],'linewidth',10) % stimulusplot(time, movmean(wheel_diff,10), 'k','linewidth',.5)
plot(time, movmean(wheel_diff,10), 'k','linewidth',.5)
set(gca,'ylim',[0 1.1],'ytick','');
ylabel('norm. movt. (wheel)') ; xlabel('time (min)')

%}




