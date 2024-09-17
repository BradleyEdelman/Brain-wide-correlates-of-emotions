function data_correct = correct_dropped_frames(fs, n, data_orig)

%% Correct dropped frames (interploate)

% Mace and Gogolla Labs - Max Planck Institute of Biological
% Intelligence/Psychiatry
% Author: Bradley Edelman
% Date: 03.11.22

% do for each column of data
data_total = cell(0);
drop_total = 0;
for i_col = 1:size(data_orig,2)
    data_orig_tmp = data_orig(:,i_col);
    
    for i_sec = 2:size(n,1)-1 % dont correct first sec (will likely not be full)
        if n(i_sec) < fs

            dropped = fs - n(i_sec);
            drop_total = drop_total + dropped;
 
            % repeat most recent frame if dropped
            before = data_orig_tmp(1:sum(n(1:i_sec)));
            repeat = repmat(data_orig_tmp(sum(n(1:i_sec))), dropped, 1);
            after = data_orig_tmp(sum(n(1:i_sec))+1:end);
            
            data_orig_tmp = cat(1, before, repeat, after);

        elseif n(i_sec) > fs

            extra = n(i_sec) - fs;
            data_orig_tmp(n(i_sec):n(i_sec)+extra-1) = [];

        end
    end

    data_total = cat(2, data_total, data_orig_tmp);
end

data_correct = data_total;

fprintf('%.0f frames dropped over %0.2f minutes...\n', drop_total, size(n,1)/60);



