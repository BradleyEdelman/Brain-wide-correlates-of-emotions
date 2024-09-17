function fus_SVD_interp(data, param)
% Author: Bradley J Edelman
% Mace and Gogolla Labs
% Max Planck Institutes for Biological Intelligence/Psychiatry

close all
proc_load = {'preprocess'};
proc_save = {'svd'};

rewrite = param.svd.rewrite;

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
                while ~exist('TMP','var'); TMP = load(proc_file{i}); end; load(proc_file{i}); clear TMP
            end
            
            % organize data format
            I_perm = permute(I_interp,[3 1 2]);
            % linearize  matrix of PD images (without mean) vs time
            I_lin = reshape(I_perm,[size(I_interp,3) nz_init*nx_init]);

            % apply svd
            [U,S,V] = svd(I_lin','econ');
            
            % variance accounted for
            f1 = figure(15); clf
            subplot(1,2,1); plot(cumsum(diag(S))./sum(diag(S)))
            xlabel('Component #'); ylabel('Cumulative VAF');
            subplot(1,2,2); bar(diag(S)./sum(diag(S)));
            xlabel('Component #'); ylabel('Individual VAF');
            title('SVD variance accounted for')

            % power spectrum of right singular vectors
            [psd,f,f2] = fus_svd_psd(V,dt_init);

            % frequency content
            stim_fidx = find(abs(f) < 0.001);
            nonstim_fidx = find(abs(f) > 0.001);
            stim_pow = mean(psd(stim_fidx,:),1);
            nonstim_pow = mean(psd(nonstim_fidx,:),1);
            figure; hold on;
            plot(stim_pow); plot(nonstim_pow)

            figure;
            for i=1:50
                [psd,f] = fft_be(V(:,i),dt_init);
                subplot(5,10,i); 
                plot(f,psd);
                title(num2str(i))
            end

            sv_fig = [save_fold 'SVD_VaF.fig']; savefig(f1, sv_fig)
            sv_png = [save_fold 'SVD_VaF.png']; saveas(f1, sv_png)
            psd_fig = [save_fold 'SVD_PSD.fig']; saveas(f2, psd_fig)
            psd_png = [save_fold 'SVD_PSD.png']; saveas(f2, psd_png)

            fprintf('\nSaving: %s\n', save_file);
            save(save_file,'U', 'S', 'V', 'psd', 't_interp','-v7.3')
            
        end
        
    end
end


