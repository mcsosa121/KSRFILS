n = 100;
k = 10;
A = rand(n,n);
A = A+A';
opts.issym = 1;
tic
    [V,et] = eigs(A,k,'lm',opts);
toc
b = randn(n,1);
tic
[W,e] = aeig(A,k,b);
toc
tic
[W2,e2] = sparse_aeig(A,k,b);
toc
norm(sort(diag(et)) - sort(diag(e)))
norm(sort(diag(et)) - sort(diag(e2)))

ortho = zeros(1,k);
ortho2 = zeros(1,k);
for i = 1:k
    v = (A-e(i,i)*eye(n,n))*W(:,i);
    ortho(i) = v'*A*A*b;
    
    v2 = (A-e(i,i)*eye(n,n))*W2(:,i);
    ortho2(i) = v2'*A*A*b;
end
ortho
ortho2