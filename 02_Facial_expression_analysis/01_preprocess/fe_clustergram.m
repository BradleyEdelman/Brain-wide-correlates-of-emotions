function [f, leafOrder] = fe_clustergram(data, labels)
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman



tree = linkage(data, 'complete');
% label = [ones(1,size(idxStim,2)) zeros(1,size(idxBase,2))];
D = pdist(data,'correlation');
leafOrder = optimalleaforder(tree,D);
img = squareform(pdist(data(leafOrder,:),'correlation'));
% img(1:size(img,1)+1:end) = max(img(:));

f = figure(7); clf; subplot(1,20,1:4);
[H,T] = dendrogram(tree,0,'Orientation','left','reorder',leafOrder);
for i_h = 1:size(H,1)
    H(i_h).Color = 'k';
    H(i_h).LineWidth = 1;
end
set(gca,'ylim',[1 size(H,1)+1]); axis off
subplot(1,20,5:19);
imagesc(img); axis xy;
colormap(magma); axis off
subplot(1,20,20);
imagesc(flipud(labels(leafOrder)')); axis off
set(gca,'colormap',[.9 .9 .9; 1 1 1])





