function fus_correct_behav_labels(data_files, param)
% Mace and Gogolla Labs - Max Planck Institute of Neurobiology
% Author: Bradley Edeman
% Date: 21.03.22

%%
rewrite = param.behavior.correct_rewrite;

if rewrite == 1

    % load behavior label meanings
    label_meanings = table2array(readtable('J:/Patrick McCarthy/behaviour_predictions/label_meanings.csv','ReadVariableNames',0));
    
    video_dir = [param.save_fold param.behavior.videos.video_fold];
    
    for i_session = 2:size(data_files,1)
        
        % specify old and new file
        current_file = data_files{i_session,2};
        corrected_file = [current_file(1:end-4) '_corrected.mat'];
        
        % load fus data (need data length and sampling rate)
        load(data_files{i_session,1});
        I_signal = reshape(I_signal, [], size(I_signal,4));
        param.fus.dt = metadata.dt;
        
        % load and downsample ORIGINAL labels
        param.behavior.labels = 'original'; % just to be sure
        labels = fus_extract_behavior_labels(I_signal, {data_files{i_session, 2:3}}, param);
        
        % extract mouse name and date for current session
        parts = strsplit(current_file,'\');
        parts = strsplit(parts{end},'_');
        mouse = parts{1};
        date = parts{2};
        
        % for each behavior, check if a bout review file (excel) exists
        % % if it does, load it and adjust labels accordingly
        for i_behavior = 1:size(label_meanings,1)

            review_file = [video_dir label_meanings{i_behavior} '\Bout_param_review.xlsx'];
            if exist(review_file,'file')

                [~, TXT, RAW] = xlsread(review_file); % load review file
                % find indices for current mouse and current day/session
                mouse_idx = find(strcmp(RAW(2:end,2), mouse) & cell2mat(cellfun(@(v) v == str2double(date), RAW(2:end,3), 'uniformoutput', false)));
        
                START = RAW(2:end, 4); % beginning of a bout (sec);
                START = cell2mat(START(mouse_idx)); % only current file
                START = round(START/.6); % convert to frames (hard code sampling rate)
                START = START - param.behavior.videos.start_adjust; % adjust bout start time according to the video adjustment

                END = RAW(2:end, 5);
                END = cell2mat(END(mouse_idx));
                END = round(END/.6);
                END = END - param.behavior.videos.end_adjust; % adjust bout end time according to the video adjustment
        
                for i_idx = 1:size(mouse_idx,1) % not very efficient but most versatile for now

                    adjust = RAW(mouse_idx(i_idx) + 1, 6);
                    if strcmp(adjust{1}, 'y') || isnan(adjust{1}) % if labeled correct
                        % do nothing
                    elseif isnumeric(adjust{1}) % adjust labels of current bout and behavior
                        labels(START(i_idx) + adjust{1}:START(i_idx)) = i_behavior;
                    else % if labeled as another behavior
                        abbrev = cellfun(@(v) v(1), label_meanings);
                        correct_behav = find(abbrev == adjust{1});
                        labels(START(i_idx):END(i_idx)) = correct_behav;
                    end
                end
            end
        end
        
        corrected_labels = labels;
        save(corrected_file, 'corrected_labels');
    end

end





