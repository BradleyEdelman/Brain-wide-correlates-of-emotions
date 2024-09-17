function [I_emotion, fUS_proto] = fus_proto_regressor_select(I, regressor, regressor_name, select, param)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% size of fUS data
sz = size(I);

% select prototypes to extract/include in the glm
regressor_tmp = zeros(min([size(select,2) size(regressor,1)]), size(regressor,2));
for i_select = 1:size(select,2)
    idx = find(strcmp(select{i_select}, regressor_name));
    if ~isempty(idx) && 0 < idx && idx <= size(regressor,1)
        regressor_tmp(i_select,:) = regressor(idx,:);
    end
end
regressor = regressor_tmp;

% convert emotion prototypes to fUS temporal sampling
if size(regressor,2) > sz(end)*5
    [regressor, regressor_norm, num_vid2fus_frame] = fus_video_to_fus(regressor, param);
else
    regressor_norm = rescale(regressor, 'InputMin', min(regressor,[],2), 'InputMax', max(regressor,[],2));
    num_vid2fus_frame = size(regressor,2);
end

cut_length = min([num_vid2fus_frame, sz(end)]);

% create output variable
fUS_proto.regressor = regressor(:,1:cut_length);
fUS_proto.regressor_norm = regressor_norm(:,1:cut_length);

% clip fUS data
if size(sz,2) == 3
    I_emotion = I(:,:,1:cut_length);
elseif size(sz,2) == 4
    I_emotion = I(:,:,:,1:cut_length);
end

