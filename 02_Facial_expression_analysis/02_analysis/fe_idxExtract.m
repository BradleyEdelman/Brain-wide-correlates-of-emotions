function [idxStim, idxBase, idxExtend] = fe_idxExtract(first_stim, stim_interval, num_stim, Lstim, Lbase)
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman


idx = first_stim:stim_interval:20*first_stim;
idx = idx(1:num_stim);

idxStim = []; idxBase = []; idxExtend = [];
for i_stim = 1:num_stim
    idxStim = [idxStim idx(i_stim):idx(i_stim) + Lstim - 1];
    idxBase = [idxBase [idx(i_stim) - Lbase:idx(i_stim) - 1]];
    idxExtend = [idxExtend idx(i_stim) - Lbase:idx(i_stim) + 10*Lstim];
end