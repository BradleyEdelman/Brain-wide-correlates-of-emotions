function Im4 = fus_2d_to_3d(testIm,fact)

[nz,nx,nt]=size(testIm);
if nargin == 1
    fact = 4;
end

if nt==1
    Im2=reshape(testIm,[nz nx/fact fact]);
    Im3=permute(Im2,[1 3 2]);
    Im4=reshape(Im3,[nz*fact nx/fact]);

else
    
    Im2=reshape(testIm,[nz nx/fact fact nt]);
    Im3=permute(Im2,[1 3 2 4]);
    Im4=reshape(Im3,[nz*fact nx/fact nt]);

end


