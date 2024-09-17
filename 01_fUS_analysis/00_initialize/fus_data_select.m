function data = fus_data_select(raw_fold, mouse_id)

for i_mouse = 1:size(mouse_id,2)
    
    data.mouse(i_mouse).id = mouse_id{i_mouse};

    info_file = [raw_fold.fus mouse_id{i_mouse} '\info\' mouse_id{i_mouse} '_journal.txt'];
    if ~exist(info_file,'file')
        info_file = [raw_fold.fus mouse_id{i_mouse} '\info\journal.txt'];
    end
	fid = fopen(info_file,'r');

    tline = fgetl(fid);
    tlines = cell(0,1);
    while ischar(tline)
        tlines{end+1,1} = tline;
        tline = fgetl(fid);
    end
    fclose(fid);

    stim = {'sucrose', 'Sucrose', 'quinine', 'Quinine', 'tail shock', 'Tail Shock', 'Tail shock'};
    fus_tot = cell(0);
    fus_tot_label = cell(0);
    fus_tot_stim_sequence = cell(0);
    
    % determine stimulus type for each run
    for i_stim = 1:size(stim,2)
        run_stim = find(contains(tlines,stim{1,i_stim}));
        run_fus = find(contains(tlines,'FUS'));
        run_fus = run_fus(cellfun(@(v) isempty(strfind(v,'(')),tlines(run_fus)));
        run_fus = run_fus(cellfun(@(v) isempty(strfind(v,'+')),tlines(run_fus)));
        run_sequence = find(contains(tlines,'Stim timing')); % see if specific timing input
        
        for i_run = 1:size(run_stim,1)
            
            % fus run options
            idx_fus_options = find(run_fus > run_stim(i_run));
            
            % stim run options
            idx_sequence_options = [];
            if ~isempty(run_sequence)
                idx_sequence_options = find(run_fus > run_sequence(i_run) & run_fus > run_stim(i_run));
            end
            
            if ~isempty(idx_fus_options)
                % id of fus data file
                idx_fus_choice = idx_fus_options(1);
                fus_tmp = tlines{run_fus(idx_fus_choice)};
                space = find(isspace(fus_tmp));
                fus_tmp = fus_tmp(space(1) + 1:space(2)-1);
                
                % id of stimulus sequence
                if ~isempty(idx_sequence_options)
                    idx_sequence_choice = idx_sequence_options(1);
                    sequence_tmp = tlines{run_sequence(idx_sequence_choice)};
                    sequence_tmp =  regexp(sequence_tmp,'\d*','Match');
                    sequence_tmp = str2double(sequence_tmp{:});

                    % custom pseudo random stim sequences
                    if sequence_tmp == 0
                        stim_sequence{1} = 180 +[0:6]*(2*60+2); % % onset (sec)
                        stim_sequence{2} = 2; % duration (sec)
                        stim_sequence{3} = 3; % lag (frames)
                    elseif sequence_tmp == 1
                        stim_sequence{1} = [180 250 340 400 510 590 690]; % onset (sec)
                        stim_sequence{2} = 2; % duration (sec)
                        stim_sequence{3} = 0; % lag (frames)
                    elseif sequence_tmp == 2
                        stim_sequence{1} = [180 280 360 470 530 620 690]; % onset (sec)
                        stim_sequence{2} = 2; % duration (sec)
                        stim_sequence{3} = 0; % lag (frames)
                    end

                else
                    sequence_tmp = 0;
                    stim_sequence{1} = 180 +[0:6]*(2*60+2); % % onset (sec)
                    stim_sequence{2} = 2; % duration (sec)
                    stim_sequence{3} = 3; % lag (frames)
                end
                
                fus_tot{end+1} = fus_tmp;
                fus_tot_label{end+1} = stim{i_stim};
                fus_tot_stim_sequence{end+1} = stim_sequence;
                
            end
        end

    end

    data.mouse(i_mouse).run = fus_tot;
    data.mouse(i_mouse).label = fus_tot_label;
    data.mouse(i_mouse).stim_sequence = fus_tot_stim_sequence;
    
    % create excel file for visual inspection
    file_name = [raw_fold.fus data.mouse(i_mouse).id '\fus\visual_inspection.xlsx'];
    if ~exist(file_name,'file')
        
        xlswrite(file_name, [{'file name', 'file label','visual inspection approval', 'notes'};...
            fus_tot', fus_tot_label', cell(size(fus_tot,2),1), cell(size(fus_tot,2),1)])
    else
        
        % if new files have been added, add them to the inspection file but
        % dont override previous notes
        [num,txt,raw] = xlsread(file_name);
        txt_run = txt(2:end,1);
        for i_run = 1:size(fus_tot',1)
            if ~ismember(fus_tot{i_run},txt_run)
                txt{end+1,1} = fus_tot{i_run};
                txt{end,2} = fus_tot_label{i_run};
            end
        end
        txt = txt(2:end,:);
        
        xlswrite(file_name, [{'file name', 'file label','visual inspection approval', 'notes'};...
            txt(:,1), txt(:,2), txt(:,3), txt(:,4)]);
        
    end
    

end

        
        

