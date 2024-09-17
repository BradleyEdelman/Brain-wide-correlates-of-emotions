function [psd,f,f1] = fus_svd_psd(V,fs)

% V are the right singular values of the SVD decomposition

nfft = 2^nextpow2(size(V,1));
psd = zeros(nfft,size(V,2));
for l=1:size(V,2)
    [pxx,f]=periodogram(V(:,l),tukeywin(size(V,1),.2),'centered',nfft,fs);
    psd(:,l)=10*log10(pxx);
end
f1 = figure(1000); clf; imagesc(1:size(V,1),f,psd);
caxis([-50 -20]); colormap jet
ylabel('Component #'); xlabel('Frequency (Hz)');
title('SVD component frequency content')
