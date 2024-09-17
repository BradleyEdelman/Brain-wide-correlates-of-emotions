function glmInfo = fus_glm(I, c, D, beta_hat_pre, mask)

% run on only brain voxels, if possible
mask = fus_3d_to_2d(mask,4);
mask = reshape(mask, [], 1);
idx_br = find(mask == 1);

% t stat degrees of freedom (# independent samples)
DF = size(I,2) - 2;

% Define t-ditributions
tdist2T = @(t,DF) (1-betainc(DF/(DF+t^2),DF/2,0.5)); % 2-tailed

% single animal fixed effect analysis
b_val = zeros(size(I,1),size(c,2),size(c,1));
t_stat = zeros(size(I,1),size(c,1));
Var_e = zeros(size(I,1),size(c,1));
residual = zeros(size(I,1),size(c,1));
p2tail = ones(size(I,1),size(c,1));

% regularize if necessary

for i_c = 1:size(c,1)
    for i = 1:size(idx_br, 1)
%         fprintf([num2str(i) ' '])
        Y = I(idx_br(i),:)';

        beta_hat = beta_hat_pre*Y;
        b_val(idx_br(i),:,i_c) = reshape(beta_hat,[1 size(c(i_c,:)',1)]);

        Var_e(idx_br(i),i_c) = (Y-D*beta_hat)'*(Y-D*beta_hat)/DF;
        
        residual(idx_br(i),i_c) = norm(Y-D*beta_hat,2);

        %Hypothesis testing; Compute the t statistic
        t_stat(idx_br(i),i_c)=c(i_c,:)*beta_hat/sqrt(Var_e(idx_br(i))*c(i_c,:)*inv(D'*D)*c(i_c,:)');
        
        if isnan(t_stat(i,i_c))
            p2tail(idx_br(i),i_c) = 1;
        else
            p2tail(idx_br(i),i_c) = 1 - tdist2T(t_stat(idx_br(i),i_c),DF);
        end
        
    end
    
end
cont = squeeze(sum(repmat(reshape(c,1,[],size(c,1)),size(I,1),1).*b_val,2));

glmInfo.b_val = b_val;
glmInfo.t_stat = t_stat;
glmInfo.Var_e = Var_e;
glmInfo.residual = residual;
glmInfo.p2tail = p2tail;
glmInfo.cont = cont;