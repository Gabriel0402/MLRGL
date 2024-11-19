function acc = DA_LPP_MV_GLR(pre_feat_S,pre_lb_S,pre_feat_T,pre_lb_T,so_feat_S,so_lb_S, so_feat_T,so_lb_T, pseudo_feat_S, pseudo_lb_S, pseudo_feat_T, pseudo_lb_T)
options.NeighborMode='KNN';
options.WeightMode = 'HeatKernel';
options.k = 30;
options.t = 1;
options.ReducedDim = 128;
options.alpha = 1;
num_class = length(unique(pre_lb_S));
pretrain_features = [pre_feat_S;pre_feat_T];
so_features = [so_feat_S;so_feat_T];
pseudo_features = [pseudo_feat_S;pseudo_feat_T];
gt = [pre_lb_S,pre_lb_T]';
X{1}=pretrain_features;
X{2}=so_features;
X{3}=pseudo_features;
opts.mu=10;
lambda1=0.9;
lambda3=0.3;
opts.lambda=[lambda1 (1-lambda1) lambda3];
opts.noisy=true;
lambda =0.02;
beta = 0.5;  
alpha = 6;    
% W=centroid_MLRSSC(X,opts);
[W,restL]=TMVC(X,lambda, beta, alpha ,gt);

P = LPP(pretrain_features,W,options);
proj_S_pre = pre_feat_S *P;
proj_T_pre = pre_feat_T *P;
P = LPP(so_features,W,options);
proj_S_source = so_feat_S *P;
proj_T_source = so_feat_T *P;
P = LPP(pseudo_features,W,options);
proj_S_pseudo = pseudo_feat_S *P;
proj_T_pseudo = pseudo_feat_T *P;
[dist1,dist4] = distmeans(proj_S_pre,proj_T_pre,pre_lb_S,num_class);
[dist2,dist5] = distmeans(proj_S_source,proj_T_source,so_lb_S,num_class);
[dist3,dist6] = distmeans(proj_S_pseudo,proj_T_pseudo,pseudo_lb_S, num_class);
% distClassMeans = (dist3+dist1)/2;
% distClusterMeans = (dist6+dist4)/2;
distClassMeans = (dist2+dist1+dist3)/3;
distClusterMeans = (dist5+dist4+dist6)/3;
expMatrix = exp(-distClassMeans);
expMatrix2 = exp(-distClusterMeans);
probMatrix = expMatrix./repmat(sum(expMatrix,2),[1 num_class]);
probMatrix2 = expMatrix2./repmat(sum(expMatrix2,2),[1 num_class]);
probMatrix = max(probMatrix,probMatrix2);
[prob,predLabels] = max(probMatrix');

% 生成时间戳，格式为 'YYYYMMDD_HHMMSS'
timestamp = datestr(now, 'yyyymmdd_HHMMSS');

% 创建带有时间戳的文件名
filename = ['result_' timestamp '.mat'];

% 保存变量到这个文件中
save(filename, 'predLabels', 'pre_lb_T');

acc = sum(predLabels==pre_lb_T)/length(pre_lb_T);
fprintf('Acc:%0.3f\n', acc);
% 
% for i = 1:num_class
%     acc_per_class(i) = sum((predLabels == pre_lb_T).*(pre_lb_T==i))/sum(pre_lb_T==i);
%     fprintf('Label:%d, Acc:%0.3f\n', i, acc_per_class(i));
% end

    
    