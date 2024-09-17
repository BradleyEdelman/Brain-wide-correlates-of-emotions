function [CC, idxStim, idxBase, idxExtend] = fe_arrange_prototype_info(proto_file, order)
% Mouse facial expression analysis package
% Stable version dated April 12, 2022
% Author: Bradley Edelman



while ~exist('TMP','var'); TMP = load(proto_file); end; load(proto_file); clear TMP
% check if video got cut short for some reason (before last trial)
if size(CC,2) > 15e3
    
    for i_proto = 1:size(order,2)
        new_idx = find(strcmp(proto_name, order{i_proto}));
        if ~isempty(new_idx)
            CC_tmp(i_proto,:) = CC(new_idx,:);
            idxStim_tmp{i_proto} = idxStim{new_idx};
            idxBase_tmp{i_proto} = idxBase{new_idx};
            idxExtend_tmp{i_proto} = idxExtend{new_idx};
        else
            CC_tmp(i_proto,:) = nan(1,size(CC,2));
            idxStim_tmp{i_proto} = nan;
            idxBase_tmp{i_proto} = nan;
            idxExtend_tmp{i_proto} = nan;
        end
    end

    % % % Old version 20210118, test new one with original stimuli
    % arrange prototypes "correctly"
%     for i_proto = 1:size(CC,1)
%         new_idx = find(strcmp(proto_name, order{i_proto}));
%         if ~isempty(new_idx)
%             CC_tmp(i_proto,:) = CC(new_idx,:);
%             idxStim_tmp{i_proto} = idxStim{new_idx};
%             idxBase_tmp{i_proto} = idxBase{new_idx};
%             idxExtend_tmp{i_proto} = idxExtend{new_idx};
%         else
%             CC_tmp(i_proto,:) = nan(1,size(CC,2));
%             idxStim_tmp{i_proto} = nan;
%             idxBase_tmp{i_proto} = nan;
%             idxExtend_tmp{i_proto} = nan;
%         end
%     end

    CC = CC_tmp; idxStim = idxStim_tmp; idxBase = idxBase_tmp; idxExtend = idxExtend_tmp;

else

    CC = []; idxStim = []; idxBase = []; idxExtend = [];

end