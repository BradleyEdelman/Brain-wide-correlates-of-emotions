function fus_load(data, param)
% Author: Bradley J Edelman
% Mace and Gogolla Labs
% Max Planck Institutes for Biological Intelligence/Psychiatry

close all
proc_save = {'preprocess'};

rewrite = param.load.rewrite;

for i_mouse = 1:size(data.mouse,2)
    for i_run = 1:size(data.mouse(i_mouse).run,2)
        
        % storage locations
        storage = fullfile([data.raw_fold data.mouse(i_mouse).id '\fus\' data.mouse(i_mouse).run{i_run} '\']);
        save_file = [storage char(proc_save) '\I_' char(proc_save) '.mat'];
        
        if exist(save_file,'file') && rewrite == 1 || ~exist(save_file,'file')
            
            % check if necessary files/folders exist
            fus_check(storage, [], proc_save);

            % load raw data file
            raw_file = [data.raw_fold data.mouse(i_mouse).id '\fus\' data.mouse(i_mouse).run{i_run} '.mat'];

            if exist(raw_file,'file')
                
                while ~exist('TMP','var'); TMP = load(raw_file); end; load(raw_file); clear TMP
                
                % adjust image if different from older/standard data (want each volume = 36x64x27)
                sz_tmp = size(I);
                if size(sz_tmp,2) == 4 % new
                    I = I(1:end-3,:,1:end-3,:); % remove dead voxels/slices
                    I = reshape(I,size(I,1),[],size(I,4)); % convert to 3D matrix
                    
                    dt_interp = 0.3; % new acquisition slightly faster, sample to something compatible with old sampling
                    sz_orig = [67 64 51];
                else
                    dt_interp = 0.6;
                    sz_orig = [36 64 27];
                end
                
                % reference image
                A = squeeze(mean(mean(I,1),2));
                [BB,II] = sort(A,'ascend');
                I_ref = real(mean(I(:,:,II(1:50)),3).^.25);
%                 I_ref = real(mean(I(:,:,100),3).^.25);
                I0 = fus_2d_to_3d(I_ref,4);
                f = figure(50); imagesc(I0);
                caxis([20 75]); axis off
                [nz_disp, nx_disp] = size(I0);

                stim_fig = [storage char(proc_save) '\brain_ref.fig']; savefig(f,stim_fig)
                stim_png = [storage char(proc_save) '\brain_ref.png']; saveas(f,stim_png)

                % size/time param
                time_init = time;
                dt_init = mean(diff(time_init));
                [nz_init, nx_init, nt_init] = size(I);
                
                % relative signal
                I_rel = relative_signal(I,1:nt_init);

                % temporal interpolation for set sampling rate
                [I_interp, t_interp] = fus_interp(I_rel, time_init, dt_interp);
                
                if dt_interp == 0.3
                    I_interp = I_interp(:,:,1:2:end);
                    t_interp = t_interp(1:2:end);
                    dt_interp = 0.6;
                end
                
                % save nifti file for manual brain masking
                I_nii = reshape(I_ref, sz_orig);
                nii = make_nii(flip(rot90(I_nii),2),[.1 .1 .1]);
                nii_file =[storage char(proc_save) '\Brain.nii'];
                save_nii(nii, nii_file)

                fprintf('\nSaving: %s\n', save_file);
                save(save_file,'I','I_ref','I0','I_rel','I_interp','t_interp','time_init','dt_init','dt_interp',...
                    'sz_orig', 'nz_init','nx_init','nt_init','nz_disp','nx_disp','-v7.3')
                
            end
            
        end
        
    end
end
        
        
        
        