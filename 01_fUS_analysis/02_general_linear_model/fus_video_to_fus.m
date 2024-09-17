function [TS_out, TS_out_norm, num_vid2fus_frame] = fus_video_to_fus(TS_in, param)


% convert time series to fUS temporal sampling
vfs = param.video.fs; % video sampling rate
fusfs = 1/(param.dt_interp); % fus sampling rate

if rem(vfs/fusfs,1) > 0.1
    % increase sampling by 10 to accommodate fraction
    TS_in = resample(TS_in,10,1);
    video_per_fus = round(vfs/fusfs*10);
    upsample = 1;
else
    video_per_fus = round(vfs/fusfs); % video frames per one fus frame
end

% ensure even number of video frames for each fus frame (cut remainder)
r = rem(size(TS_in,2),video_per_fus); % extra frames
% % % % TS_in = TS_in(:,1:end-r); % remove extra frames
if r ~= 0
    TS_in = [TS_in nan(size(TS_in,1), video_per_fus - r)]; % add nan frames to "fill in" last frame
end

% total number of fUS frames to keep (video is limiting factor)
num_vid2fus_frame = size(TS_in,2)/video_per_fus;

% bin prototypes into fUS termporal sampling according to above parameters
TS_in = reshape(TS_in', video_per_fus, [], size(TS_in,1));
TS_out = squeeze(nanmean(TS_in,1));

% make sure its a long matrix
if size(TS_out,1) > size(TS_out,2)
    TS_out = TS_out';
end

% min-max normalize
TS_out_norm = rescale(TS_out, 'InputMin', min(TS_out,[],2), 'InputMax', max(TS_out,[],2));


