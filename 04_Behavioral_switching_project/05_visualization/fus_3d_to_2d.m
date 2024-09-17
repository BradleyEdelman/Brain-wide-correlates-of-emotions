function Im4 = fus_3d_to_2d(testIm, fact)

[nz,nx,nt]=size(testIm);
if nargin == 1
    fact = 4;
end

if nt==1
    
    Im2 = reshape(testIm, [nz/fact fact nx]);
    Im3 = permute(Im2, [1 3 2]);
    Im4 = reshape(Im3, [nz/fact nx*fact]);
    
else
    
    Im2 = reshape(testIm, [nz/fact fact nx nt]);
    Im3 = permute(Im2, [1 3 2 4]);
    Im4 = reshape(Im3, [nz/fact nx*fact nt]);

end