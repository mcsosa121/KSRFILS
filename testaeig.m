A = rand(10,10);
A = A+A';
[V,et] = eigs(A,3);
[W,e] = aeig(A,3);
[X,eo] = old_aeig(A,3);
sort(diag(et)) - sort(diag(e))
sort(diag(et)) - sort(diag(eo))