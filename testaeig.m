n = 100;
k = 10;
A = rand(n,n);
A = A+A';
opts.issym = 1;
tic
    [V,et] = eigs(A,k,'lm',opts);
toc
tic
[W,b,e] = aeig(A,k); %note: change function signature in aeig.m to [V,b,e] to run this
toc
norm(sort(diag(et)) - sort(diag(eo)))

orth = zeros(1,k);
for i = 1:k
    v = (A-e(i,i)*eye(n,n))*W(:,i);
    orth(i) = v'*A*A*b;
end
orth