function [S,restL]=TMVC(X,r1,w1,alpha,Y)

k = 15;
v = length(X);
n = size(X{1}, 1);
c = max(Y);
S = cell(v, 1);
[norX] = NormalizeData(X);
for v1 = 1 : v
    S{v1} = constructW_PKN(norX{v1}', k);
end

La = cell(v, 1);
H = cell(v, 1);
HH = cell(v, 1);
hatH = cell(v, 1);
hatHH = cell(v, 1);
Q = cell(v, 1);
z = cell(v, 1);
for v1 = 1:v
    DN = diag( 1./sqrt(sum(S{v1})+eps));
    La{v1} = DN * S{v1} * DN;
    H{v1} = zeros(n, c);
    HH{v1} = H{v1}*H{v1}';
    Q{v1} = zeros(n, n);
    hatH{v1} = Q{v1}*H{v1};
    hatHH{v1} = hatH{v1}*hatH{v1}';
    z{v1} = zeros(n, n);
end

w2=1-w1;
tol = 1e-4; 
max_iter = 1;
rho = 1.1;
mu = 1e-4;
max_mu = 1e10;
tnn = zeros(n, 1);
L = zeros(n,n,v);
L1 = L;
L2 = L;
L3 = L;

Lambda21=L;
Lambda22=L;
Lambda23=L;

for iter = 1 : max_iter

    for  i = 1 : v
        z{i} = L( : , : , i);
    end
    temp = cell(v, 1);
    G = cell(v, 1);
    for i = 1 : v
        temp{i} = Q{i} * ((0.5 * (z{i} + z{i}') - 0.5 * hatHH{i})) * Q{i};  
        G{i} = r1 * La{i} + temp{i};                                  
        [H{i}] = eig2(G{i}, c);
        HH{i} = H{i} * H{i}';
        Q{i} = diag(1 ./ sqrt(diag(HH{i})));
        hatH{i} = Q{i} * H{i};            
        hatHH{i} = hatH{i} * hatH{i}';    
    end

    if v == 2
        X1(:,:,1)=abs(hatHH{1})+abs(hatHH{1}');
        X1(:,:,2)=abs(hatHH{2})+abs(hatHH{2}');
    end
    
    if v == 3
        X1(:,:,1)=abs(hatHH{1})+abs(hatHH{1}');
        X1(:,:,2)=abs(hatHH{2})+abs(hatHH{2}');
        X1(:,:,3)=abs(hatHH{3})+abs(hatHH{3}');
    end

    
    Temp=(1/(1+mu))*(X1+L+(Lambda21/mu));    
    for i=1:v
        Tempvector=0.5*(Temp(:,:,i)+Temp(:,:,i)');
        L1(:,:,i)=prox_nuclear(Tempvector,w1/(1+mu));
    end
    
    Temp=(1/(1+mu))*(X1+L+(Lambda22/mu)); 

    for i=1:n
        Tempvector=Temp(:,i,:);
        Tempvector=reshape(Tempvector,n,v);
        L2(:,i,:)=reshape(prox_nuclear(Tempvector,w2/(1+mu)),n,1,v);
    end
    
    Temp=(1/(1+mu))*(X1+L+(Lambda23/mu));  
    for i=1:n
%         Tempvector=Temp(:,i,:);
%         Tempvector=reshape(Tempvector,n,v);
%         [L33,tnn(i)] = prox_l21(Tempvector,(w2*alpha)/((1+mu)));
%         L3(:,i,:)=reshape(L33,n,1,v);
            Tempvector = Temp(:, i, :);
            Tempvector = reshape(Tempvector, n, v);
            % 调试信息
            disp(['Tempvector for prox_l21 at iteration ', num2str(iter), ' and index ', num2str(i), ':']);
            disp(Tempvector);
            [L33, tnn(i)] = prox_l21(Tempvector, (w2 * alpha) / (1 + mu));
            L3(:, i, :) = reshape(L33, n, 1, v);
            % 调试信息
            disp(['tnn(', num2str(i), ') = ', num2str(tnn(i))]);
    end
   
    L=(1/3)*(L1+L2+L3-(Lambda21+Lambda22+Lambda23)/mu);

    d11=X1-L1;
    d12=X1-L2; 
    d13=X1-L3;

    d21=(L-L1);
    d22=(L-L2);
    d23=(L-L3);


    chg= max([ max(abs(d11(:))),max(abs(d12(:))),max(abs(d13(:))),max(abs(d21(:))),max(abs(d22(:))),max(abs(d23(:))) ]);
    
    disp(['#iteration: ',num2str(iter)]);
    
    if chg < tol
        break;
    end
    

    f = zeros(v, 1);
    for i = 1 : v 
        f(i) = 0.5* r1 * norm(La{i} - HH{i}, 'fro')^2 + 0.5 * norm(L(:,:,i)- hatHH{i}, 'fro') ^ 2;
    end
    
    obj(iter) = sum(f)+ sum(tnn);
    
    %fprintf('err: %0.3f:\n',obj(iter));
    if iter > 1
      fprintf('err: %0.3f:\n',obj(iter) - obj(iter - 1));
    end
    if iter > 40 && abs((obj(iter) - obj(iter - 1)) / obj(iter - 1)) < 1e-6
        break;
    end
    
    Lambda21=Lambda21+mu*d21;
    Lambda22=Lambda22+mu*d22;
    Lambda23=Lambda23+mu*d23;
    
    mu = min(rho*mu,max_mu);    
   
end


disp(['iter:', num2str(iter)]);
if v == 2    
  S=abs(L(:,:,1))+abs(L(:,:,2));
  elseif v == 3 
     S=abs(L(:,:,1))+abs(L(:,:,2))+abs(L(:,:,3));
  elseif v == 4 
     S=abs(L(:,:,1))+abs(L(:,:,2))+abs(L(:,:,3))+abs(L(:,:,4));
end   
S=S-diag(diag(S));

cls_num = length(unique(Y));
C = SpectralClusteringqi(S,cls_num);
[restL] = myNMIACC(real(C), Y, cls_num); 