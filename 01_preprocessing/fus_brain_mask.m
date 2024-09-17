function fus_brain_mask(data, param)
% Author: Bradley J Edelman
% Mace and Gogolla Labs
% Max Planck Institutes for Biological Intelligence/Psychiatry

close all
proc_load = {'preprocess'};
proc_save = {'mask'};

rewrite = param.mask.rewrite;

for i_mouse = 1:size(data.mouse,2)
    for i_run = 1:size(data.mouse(i_mouse).run,2)
        
        % storage locations
        storage = [data.raw_fold data.mouse(i_mouse).id '\fus\' data.mouse(i_mouse).run{i_run} '\'];
        save_fold = [storage char(proc_save) '\'];
        save_file = [save_fold 'I_' char(proc_save) '.mat'];
        
        if exist(save_file,'file') && rewrite == 1 || ~exist(save_file,'file')
            
            % check if necessary files/folders exist
            proc_file = fus_check(storage, proc_load, proc_save);
            for i = 1:size(proc_file,2)
                while ~exist('TMP','var'); TMP = load(proc_file{i},'I0'); end; load(proc_file{i},'I0','nz_init'); clear TMP
            end
        
            mask_file = [storage char(proc_load) '\mask.nii.gz'];
            if exist(mask_file,'file')

                nii = load_nii(mask_file);
                nii.img = fliplr(flipud(permute(nii.img, [2 1 3])));
                nii.img = reshape(nii.img, nz_init, []);
                mask = fus_2d_to_3d(nii.img);

            else

                mask = ones(size(I0));

            end

            f = figure(10); clf; [h, h2] = transparence_jet_3d(I0,[0 100],mask,[0 1],0.2,ones(size(I0)),0);

            savefig(f,[storage char(proc_save) '\mask_check.fig'])
            saveas(f,[storage char(proc_save) '\mask_check.png'])

            fprintf('\nSaving: %s\n', save_file);
            save(save_file,'mask')
        end
    end
end
        
        
        
        
        
        
        
        
        
        
        
        