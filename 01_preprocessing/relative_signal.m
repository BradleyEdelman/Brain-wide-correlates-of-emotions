function Irel=relative_signal(Itot,baseline)

%baseline = 1:size(Itot,3); % definition of the baseline in frames
Iref = squeeze(mean(Itot(:,:,baseline),3,'omitnan'));

% make relative
Irel = (Itot - repmat(Iref,[1 1 size(Itot,3)])) ./ repmat(Iref,[1 1 size(Itot,3)]);