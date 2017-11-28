A = rand(100,100);
A = A+A';
opts.issym = 1;
tic
    [V,et] = eigs(A,10,'lm',opts);
toc
tic
[X,eo] = old_aeig(A,10);
toc
tic
[Y,Z,Q,T,eoo] = space_aeig(A,10);
toc
norm(sort(diag(et)) - sort(diag(eo)))
norm(sort(diag(et)) - sort(diag(eoo)))

eigt = 0;
eigspace = 0;
eigx = 0;
for i = 1:10
    eigt = eigt+norm(A*V(:,i)-et(i,i)*V(:,i));
    eigspace = eigspace+norm(A*Y(:,i)-eoo(i,i)*Y(:,i));
    eigx = eigspace+norm(A*X(:,i)-eoo(i,i)*X(:,i));
end
eigt
eigspace
eigx