function [h, h2] = transparence_jet_3d(G,caxG,H,caxH,transp,mask,stim)

% G image in gray
% caxG caxis gray
% H image couleur
% caxH caxis couleur
% trasnp facteur transparence (matrix ou number)
% mask masque (c'est possible de le mettre direct dans la transp direct
% finalement il fait transp*mask.

disp('ok')

dz= 0.3; %mm;
dx=  0.3; %mm

[nz,nx]=size(G);
z=(0:nz-1)*dz;
x=(0:nx-1)*dx;

JET=colormap(jet(256));


% figure(90); %clf

imagesc(x,z,G);
colormap gray,
caxis(caxG);
hold on

N=length(JET);

H1=round((H-caxH(1))./(caxH(2)-caxH(1))*N);
H1(find(H1<1))=1;

[nx,nz]=size(H1);
% figure
h = image(x,z,ind2rgb(H1,JET),'AlphaData',transp*mask);
% axis equal; axis tight;

if stim == 1
    Stim = ones(size(G));
    
    % Hack for placing stim indicator on different image sizes
    if sum(size(H1) == [144 432]) == 2
        y_idx = 5:10; x_idx = 5:50;
        Stim(y_idx,x_idx) = round(max(caxG)/4*6.5);
    else
         y_idx = 5:20; x_idx = 43:183;
         Stim(y_idx,x_idx) = round(max(caxG)/3*2);
    end
    
    mask2 = zeros(size(G));
    mask2(y_idx,x_idx) = 1;
    h2 = image(x,z,ind2rgb(Stim,JET),'AlphaData',transp*mask2);
else
    h2 = [];
end
    

