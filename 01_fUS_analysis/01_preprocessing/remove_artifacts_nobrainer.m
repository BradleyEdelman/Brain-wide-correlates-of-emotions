function [I_denoised,nPC,tPC,r_sq] = remove_artifacts_nobrainer(I,brain_mask,varargin)
% REMOVE_ARTIFACTS_NOBRAINER removes motion artifacts from data (I) by
%  egressing a low-dimentional representation (top principal components) of
%  the no-brain voxels (0 at brain_mask).
% 
% FORMAT:    [I_denoised, nPC, tPC] = remove_artifacts_nobrainer(I, brain_mask, varargin)
%
% REQUIRED INPUTS:
%    I                  -   data matrix, last dimension must be temporal.
%    brain_mask         -   mask for voxels inside the brain (logical, same
%                           spatial dimensions as I). 1 represents brain, 0   
%                           represents non-brain. The SVD will be performed  
%                           on the voxels marked as 0. 
%
% OPTIONAL INPUTS:
%    'threshold_type'	-   type of the threshold to select # of regressed PCs:
%        'proportion'   -   (default) regress a proportion of all components.
%                           Also works with 'proportion' and 'length'.
%        'fixed'        -   regress a fixed number of components.
%        'variance'     -   regress the number of components that explains a 
%                           certain percentage of total variance.
%
%    'threshold_value'	-   set a custom value for the selected threshold type.
%                           Scalar between 0 and 1 for 'proportion' and
%                           'variance', positive integer for 'fixed'.
%                           Defaults are 0.05 for 'proportion', 5 for 'fixed'
%                           and 0.80 for 'variance'.
%
%    'scale'            -   Specify the scaling factor for the voxels of I.
%        'mean'         -   (default) divide by the mean per-voxel.
%        'std'          -   divide by standard deviation per-voxel.
%        'none'         -   no scaling.
%
%    'motion_extr'      -   Specify the method to extract motion components from I.
%        'out_pca'      -   (default) Principal Components Analysis of the out-brain voxels.
%        'in_out_cca'	-   Canonical Correlation Analysis between in- and out-brain voxels.
%
%    'pc2keep'          -   number of components to keep using the CCA method. If scalar
%                           between 0 and 1, it means a percentage of total components.
%                           If positive integer, it means a total number of components
%                           Default is 40% of total components.
%
% OUTPUTS:
%    I_denoised         -   denoised data matrix.
%    nPC                -   number of removed Principal Components (PCs).
%    tPC                -   regressed temporal PCs, each column corresponds 
%                           to a single tPC. 
%    r_sq               -   R2 values of the PCA-regression per voxel
%
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% EXAMPLES:           
%
%       I_denoised = REMOVE_ARTIFACTS_NOBRAINER(I, brain_mask) removes motion
%       artifacts from I by regressing the time-traces of the 5% first PCs
%   
%       [I_denoised, nPC, tPC] REMOVE_ARTIFACTS_NOBRAINER(...) return number of
%       regressed PCs and all PC time traces.
%
%       [~, ~, tPC] REMOVE_ARTIFACTS_NOBRAINER(...) return only the
%       PC time traces (e.g. to use them as nuisance regressors)
%
%
%   Optional variables can work as name-value pairs or as positional
%   variables. 
%
%   Name-value pairs:
%
%       I_denoised = REMOVE_ARTIFACTS_NOBRAINER(...,'threshold_type','variance') 
%       regress the number of PCs that explain 80% of variance.
%
%       I_denoised = REMOVE_ARTIFACTS_NOBRAINER(...,'threshold_type','fixed','threshold', 20)
%       regress the top 20PCs from total. 
%
%       I_denoised = REMOVE_ARTIFACTS_NOBRAINER(...,'scale', 'std') use zscore
%       normalization (scaled over standard deviation).
%
%       I_denoised = REMOVE_ARTIFACTS_NOBRAINER(...,'motion_extr', 'in_out_cca')
%       use canonical correlation analysis to identify common components between
%      brain and non-brain.
%
%   Positional inputs:
%
%       I_denoised = REMOVE_ARTIFACTS_NOBRAINER(..., 'fixed', 20, 'none', 'in_out_cca', 0.50) 
%       regresses the 20 first Canonical Components after centering around
%       mean without scaling and reducing dimensionality to half.
%
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% ASSUMPTIONS AND LIMITATIONS:
%   Assumes that the voxels outside the brain_mask only contain motion
%   artifacs with no strong correlation to the relevant hemodynamic signal
%
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Vladimir Bukovkin and Jose Maria Martinez de Paz, 2023


% Set defaults and parse inputs
defaultType = 'proportion';

validTypes = {'fixed', 'percent', 'proportion', 'length', 'var', 'variance'};
validScaleTypes = {'none', 'no', 'mean', 'sd', 'std', 'z'};
validApproach = {'out_pca','in_out_cca'};

checkTypes = @(x) any(validatestring(x,validTypes ));
checkScaleType = @(x) any(validatestring(x,validScaleTypes));
checkApproach = @(x) any(validatestring(x,validApproach));
checkNumber = @(x) isnumeric(x) && x >= 0;

p = inputParser();
p.addOptional('threshold_type', defaultType, checkTypes);
p.addOptional('threshold_value', -1, checkNumber);  % placeholder, not really used but avoids error in next line
p.addOptional('scale', 'std', checkScaleType);
p.addOptional('motion_extr', 'out_pca', checkApproach);
p.addOptional('pc2Keep', int16(size(I, ndims(I))*0.4), checkNumber);
p.parse(varargin{:});
in = p.Results;

if in.threshold_value == -1
    switch in.threshold_type
        case 'fixed',                               in.threshold_value = 5;
        case {'length','percent','proportion'},     in.threshold_value = 0.05;
        case {'var', 'variance'},                   in.threshold_value = 0.80;
    end
end

% Make sure data has at least 2 dimensions (voxels and timepoints)
if isvector(I) 
    error('Data matrix must at least have 2 dimensions!')
end

% Make sure that mask and data dimensions match
if isvector(brain_mask)
    size_mask = length(brain_mask);
else
    size_mask = size(brain_mask, 1:ndims(brain_mask));
end
if any(size(I, 1:ndims(I)-1) ~= size_mask)
    ME = MException('no_brainer:spatialDimensionsMismatch', ...
        sprintf('Spatial dimensions do not match between the data matrix and the brain mask.'));
    throw(ME);
end

% -------------------------------------------------------------------------
% 0. Prepare input matrix
% -------------------------------------------------------------------------
fprintf('\n\t No-brainer started\n');
tic

% Linearise input matrix
I_lin = reshape(I, [], size(I, ndims(I)));  % [n_voxels x n_timepoints]

% Center data (requirement for PCA)
fprintf('\t\t-> Centering data around mean...\n');
I_mus = mean(I_lin, 2, 'omitnan');
I_norm = I_lin - I_mus;

% Normalize time-traces
switch in.scale
    case {'none', 'no'}
        fprintf('\t\t-> Data is not scaled.\n');
    case {'mean'}
        fprintf('\t\t-> Scaling over mean (%% of mean)...\n');
        I_norm = I_norm ./ I_mus;
    case {'sd', 'std', 'z'}
        fprintf('\t\t-> Scaling over standard deviation (z-score)...\n');
        I_sigma = std(I_norm, 0, 2, 'omitnan');
        I_norm = I_norm ./ I_sigma;
end
I_norm(isnan(I_norm)) = 0; 

% Transpose to the [n_obs, n_features] convention => [nt, n_voxels]
I_norm = I_norm';

% Exclude all voxels outside the registered space (nan or 0 in the first frame)
reg_mask = ~( isnan(I_lin(:,1)) | (I_lin(:,1) == 0) ); 
nobrain_lin = ~brain_mask(:) & reg_mask(:);

% -------------------------------------------------------------------------
% 1. Extract motion components
% -------------------------------------------------------------------------
fprintf('\t\t-> Extracting motion components using %s...\n', in.motion_extr);
switch(in.motion_extr)
    case 'out_pca'
        % Singular value decomposition
        [U,S,~] = svd(I_norm(:,nobrain_lin), 'econ');
 
        % Reconstruct temporal components 
        tPC = U*S;  % [nt, n_nobrain_voxels] -> (n_nobrain_voxels = nt because of 'econ')
 
    case 'in_out_cca'
        % Whiten in-brain
        [Uin,~,Vin] = svd(I_norm(:,brain_mask(:)), 'econ');
        Din_w = Uin(:,1:in.pc2Keep)*Vin(:,1:in.pc2Keep)';
        
        % Whiten out-brain
        [Uout,~,Vout] = svd(I_norm(:,nobrain_lin), 'econ');
        if in.pc2Keep < 1, in.pc2Keep = size(Uout,2) * in.pc2Keep; end
        Dout_w = Uout(:,1:in.pc2Keep)*Vout(:,1:in.pc2Keep)';
        
        % SVD of concatenated data (canonical correlation analysis)
        Dcat = [Din_w, Dout_w];
        [tPC,S,~] = svd(Dcat,'econ');
end

% Enforce a sign convention on the coefficients -- the largest element in each column will have a positive sign
[~,maxind] = max(abs(tPC), [], 1);
[d1, d2] = size(tPC);
colsign = sign(tPC(maxind + (0:d1:(d2-1)*d1)));
tPC = tPC.*colsign;
 
% -------------------------------------------------------------------------
% 2. Find nPC threshold
% -------------------------------------------------------------------------
evals = S.^2 / (nnz(nobrain_lin)-1) ;   % estimate eigenvalues from singular values
explained_var = diag(evals)/sum(diag(evals)) ;
switch in.threshold_type
    
    case 'fixed'
        fprintf('\t\t-> Selecting %i components (fixed threshold)...\n', in.threshold_value);
        nPC = int16(min(size(tPC,2), in.threshold_value));
        
    case {'length','percent','proportion'}
        fprintf('\t\t-> Selecting number of components (proportion threshold, %.2f%%)...\n', in.threshold_value*100);
        if in.threshold_value > 1, error('Percentage must be expressed as a decimal number between 0 and 1!'); end
        nPC = int16(size(tPC,2)*in.threshold_value);
        
    case {'var', 'variance'}
        fprintf('\t\t-> Selecting number of components (variance threshold, %.2f%% of total variance)...\n', in.threshold_value*100);
        if in.threshold_value > 1, error('Percentage must be expressed as a decimal number between 0 and 1!'); end
        nPC = find(cumsum(explained_var) > in.threshold_value, 1, 'first');
end

% -------------------------------------------------------------------------
% 3. Regress out nPC components
% -------------------------------------------------------------------------
fprintf('\t\t-> Regressing the first %i components out of %i (accounting for %.3f%% of total variance)...\n', nPC, size(tPC,2), 100*sum(explained_var(1:nPC)));

% Get residuals from linear model (denoised data)
X = zscore(tPC(:,1:nPC),0,1);   % [t x n_regressors]
B = (X'*X) \ (X'*I_norm);       % [n_regressors x n_regressors] \ [n_regressors x n_variables] = [n_regressors x n_variables]
I_denoised = I_norm - X*B;      % [t x n_variables] - [t x n_variables]
  
% Get goodness-of-fit (R^2)
r_sq = 1  -  sum(I_denoised.^2) ./ sum( I_norm.^2 );

% -------------------------------------------------------------------------
% 4. Recover original shape and scale
% -------------------------------------------------------------------------
% Transpose back to [n_voxels, nt]
I_denoised = I_denoised' ;

% Undo the voxel-wise normalization
switch in.scale
    case {'mean'}
        I_denoised = I_denoised .* I_mus; 
    case {'sd', 'std', 'z'}
        I_denoised = I_denoised .* I_sigma; 
end
I_denoised = I_denoised + I_mus; 
        
% Recover original matrix shape
if ~ismatrix(I)
    I_denoised = reshape(I_denoised, size(I));
    r_sq = reshape(r_sq, size(I,1:ndims(I)-1));
end

ElapsedTime = toc;
fprintf('\t\t-> Done! Elapsed time: %g seconds \n', ElapsedTime);

end
