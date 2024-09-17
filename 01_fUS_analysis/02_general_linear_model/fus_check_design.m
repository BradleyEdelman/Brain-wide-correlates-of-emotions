function [D, beta_hat_pre, fail_flag] = fus_check_design(stim)

% design matrix should be tall also
sz1 = size(stim);
if sz1(1) < sz1(2); stim = stim'; end
stim(:,end+1) = ones(1,max(sz1));

D = stim;

% check condition of design matrix, regularize if needed
if rcond(inv(D'*D)) < 1e-15
    fprintf('\nRegularizing Design Matrix...\n')
    lambda = .1;
    beta_hat_pre = inv(D'*D + eye(size(D,2))*lambda)*D';
    fail_flag = 1;
else
    beta_hat_pre = inv(D'*D)*D';
    fail_flag = 0;
end