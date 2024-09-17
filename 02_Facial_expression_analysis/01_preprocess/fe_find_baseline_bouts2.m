function [stim, base, extend] = fe_find_baseline_bouts2(CC, proto_name, param)
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman

Lbase = param.hog_disp.lbase;
Lstim = param.hog_disp.lstim;
numtrial = param.hog_disp.numtrial;

%%

CC_neutral = CC(find(strcmp(proto_name, 'neutral')),:);

Ltrial = Lbase + 4*Lstim;
idx = 2:Ltrial; cnt = 1;
while idx(end) <= 3500
    win_idx{cnt} = idx;
    ave_act(cnt) = mean(zscore(CC_neutral(idx)));
    idx = idx + 1;
    cnt = cnt + 1;
end

[B,I] = sort(ave_act,'descend');
clear total_bouts win_select
for i_bout = 1:numtrial
	total_bouts{i_bout} = win_idx{I(1)};
    win_select(i_bout) = I(1);
    
    Contain = find(sum(ismember(cat(1,win_idx{:}),cat(2,total_bouts{:})),2) > 0);
    I(find(ismember(I,Contain))) = [];
    B(find(ismember(I,Contain))) = [];
end

clear base stim baseline_bouts baseline_bouts_baseline
for i_bout = 1:size(total_bouts,2)
    baseline_bouts_baseline{i_bout} = total_bouts{i_bout}(1:Lbase);
    baseline_bouts{i_bout} = total_bouts{i_bout}(Lbase + 1: Lbase + Lstim);
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
time = (1:size(CC_neutral,2))/20/60;
plot([time(idxBase); time(idxBase)],[min(CC_neutral) max(CC_neutral)],'color',[.75 .75 .75]) % baseline
plot([time(idxStim); time(idxStim)],[min(CC_neutral) max(CC_neutral)],'color','c') % stimulus
plot(time, CC_neutral, 'r')
plot([time(3600:3640); time(3600:3640)],[min(CC_neutral) max(CC_neutral)],'color','g') % stimulus


%% plotting a bit for verification
figure(25); clf
Y = [-.1 .35];
for i = 1:size(CC,1)
    subplot(1,size(CC,1),i); hold on

    tmp_base = CC(i,idxBase); tmp_base = reshape(tmp_base',[],size(idxBase,2)/80);
    mean_base = mean(tmp_base,1);
    tmp_stim = CC(i,idxStim); tmp_stim = reshape(tmp_stim',[],size(idxStim,2)/40);
    % remove baseline
    tmp_tot = [tmp_base; tmp_stim];
    tmp_tot = tmp_tot - repmat(mean_base,size(tmp_tot,1),1);
    tmp_tot(tmp_tot < 0) = 0;

    plot(tmp_tot)

    plot(max(tmp_tot,[],2) + .1)

    plot(max(tmp_tot,[],2) + .2)
    
    set(gca,'ylim',Y,'ytick','')
    title(proto_name{i})
end
subplot(1,size(CC,1),1); set(gca,'ytick',linspace(Y(1),Y(2),7))



