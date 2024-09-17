function I_svd = fus_svd_recon(U,S,V,nfilt)


Sf=S;
Sf(1:nfilt,1:nfilt)=0;
I_svd=U*Sf*V';

