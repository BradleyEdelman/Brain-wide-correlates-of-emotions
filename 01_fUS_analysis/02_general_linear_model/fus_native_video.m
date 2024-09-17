function fus_native_video(data, param)

close all
proc_load = {'preprocess', 'mask', 'svd_removal'};
proc_save = {'video'};

rewrite = param.video.std.rewrite;

for i_mouse = 1:size(data.mouse,2)
    for i_run = 1:size(data.mouse(i_mouse).run,2)
        
        % storage locations
        storage = [data.raw_fold data.mouse(i_mouse).id '\fus\' data.mouse(i_mouse).run{i_run} '\'];
        save_fold = [storage char(proc_save) '\'];
        save_file = [save_fold 'standard_video.avi'];
        
        if exist(save_file,'file') && rewrite == 1 || ~exist(save_file,'file')
            
            % check if necessary files/folders exist
            proc_file = fus_check(storage, proc_load, proc_save);
            proc_file = proc_file(cellfun(@(v) ~contains(v,'NEW_FORMAT'), proc_file));
            for i = 1:size(proc_file,1)
                load(proc_file{i});
            end
            
            % reference image
            I_ref = fus_2d_to_3d(I_ref);
            
            if prod(size(I_ref,1:2)) ~= 62208
                % convert "new" slice/volume size to "old" size
                old_sz = [36 64 27]; % hardcoded
                new_sz = [67 64 51]; % hardcoded

                % define interpolation grid
                [x,y,z] = meshgrid(1:new_sz(2), 1:new_sz(1), 1:new_sz(3));
                [x1,y1,z1] = meshgrid(linspace(1,new_sz(2),old_sz(2)),...
                    linspace(1,new_sz(1),old_sz(1)), linspace(1,new_sz(3),old_sz(3)));
                
                I_ref = fus_3d_to_2d(I_ref,4);
                I_ref = reshape(I_ref,new_sz);
                I_ref = interp3(x,y,z,I_ref,x1,y1,z1);
                I_ref = reshape(I_ref,old_sz(1),[]);
                I_ref = fus_2d_to_3d(I_ref);
            end
            
            % functional data
            mask = double(mask);
            
            % stim
            stim_list = ceil(data.mouse(i_mouse).stim_sequence{i_run}{1}./.6);
            
            clear t
            for i_stim = 1:size(stim_list,2)
                tmp = I_svd_removal_mean(:,:,stim_list(i_stim) - 49:stim_list(i_stim) + 50);
                tmp = fus_2d_to_3d(tmp,4);
                t(:,:,:,i_stim) = tmp;
            end
            t = mean(t,4);
            
            save_file = [save_fold 'standard_video.avi'];

            I_data = {movmean(t,5,3)};
            I_name = {getName(I_svd_removal_mean)};

            fus_create_video(save_file, I_data, I_name, I_ref, mask, [-.25 .25], param);
        end
    end
end

end


function out = getName(var)
    out = inputname(1);
end

