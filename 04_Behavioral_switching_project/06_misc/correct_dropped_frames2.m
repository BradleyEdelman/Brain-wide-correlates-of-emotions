function data_correct = correct_dropped_frames2(fs, n, data_orig, flag_correct)

%% Correct dropped frames (interploate)

% Mace and Gogolla Labs - Max Planck Institute of Biological
% Intelligence/Psychiatry
% Author: Bradley Edelman
% Date: 03.11.22

if isempty(n)
    total_sec = 0;
else
    total_sec = size(n,1);
    if n(1) + n(end) <= fs; total_sec = total_sec - 1; end
end

if flag_correct == 1
    % do for each column of data
    data_total = cell(0);
    drop_total = 0;
    for i_col = 1:size(data_orig,2)
        data_orig_tmp = data_orig(:,i_col);
        
        for i_sec = 2:size(n,1)-1 % dont correct first sec (will likely not be full)
            if n(i_sec) < fs - 1
                
                dropped = fs - n(i_sec);
    %             if dropped ~= n(i_sec + 1) - fs
                    drop_total = drop_total + dropped;
    
                    % repeat most recent frame if dropped
                    before = data_orig_tmp(1:sum(n(1:i_sec)));
                    repeat = repmat(data_orig_tmp(sum(n(1:i_sec))), dropped, 1);
                    after = data_orig_tmp(sum(n(1:i_sec))+1:end);
    
                    data_orig_tmp = cat(1, before, repeat, after);
                    
    %             end
    
            elseif n(i_sec) > fs + 1
                
                extra = n(i_sec) - fs;
                data_orig_tmp(sum(n(1:i_sec)) + 1:sum(n(1:i_sec)) + extra) = [];
    
            end
        end
    
        data_total = cat(2, data_total, data_orig_tmp);
    end

    fprintf('%.0f/%0.f frames detected over %.2f min...correcting dropped frames, creating video...\n', size(data_orig,1), total_sec*fs, total_sec/60)
    
%     fprintf('%.0f frames dropped over %0.2f minutes...\n', drop_total, size(n,1)/60);

elseif flag_correct == 0

    fprintf('%.0f/%0.f frames detected over %.2f min...no correction, creating video...\n', size(data_orig,1), total_sec*fs, total_sec/60)
    data_total = data_orig;

elseif flag_correct == -1
    
    fprintf('%.0f/%0.f frames detected over %.2f min...no correction, no video...\n', size(data_orig,1), total_sec*fs, total_sec/60)
    data_total = [];

end
data_correct = data_total;





