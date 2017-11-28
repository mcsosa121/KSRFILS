n = 100;
k = 10;
A = rand(n,n);
A = A+A';
opts.issym = 1;
tic
    [V,et] = eigs(A,k,'lm',opts);
toc
tic
b = randn(n,1);
[W,e] = aeig(A,k,b); %note: change function signature in aeig.m to [V,b,e] to run this
toc
norm(sort(diag(et)) - sort(diag(eo)))

ortho = zeros(1,k);
for i = 1:k
    v = (A-e(i,i)*eye(n,n))*W(:,i);
    ortho(i) = v'*A*A*b;
end
ortho