function [I_interp, t_interp] = fus_interp(I, time, new_dt)

t_interp=new_dt:new_dt:time(end);
nt_interp=size(t_interp(:),1);
[nz, nx, nt] = size(I);

I_interp=zeros(nz,nx,nt_interp);
parfor (iz=1:nz, 6)
    disp(['Interp slice # ' num2str(iz)])
    for ix=1:nx
        I_interp(iz,ix,:)=interp1(time,squeeze(I(iz,ix,:)),t_interp,'nearest');
    end
end

I_interp = remove_NaN(I_interp);

