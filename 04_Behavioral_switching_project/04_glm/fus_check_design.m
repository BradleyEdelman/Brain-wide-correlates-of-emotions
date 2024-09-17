function [D, beta_hat_pre] = fus_check_design(stim)

% design matrix should be tall also
sz1 = size(stim);
if sz1(1) < sz1(2); stim = stim'; end
stim(:,end+1) = ones(1,max(sz1));

D = stim;

% check condition of design matrix, regularize if needed
if cond(inv(D'*D)) < -1e15
    lambda = 5;
    beta_hat_pre = inv(D'*D + eye(size(D,2))*lambda)*D';
else
    beta_hat_pre = inv(D'*D)*D';
end