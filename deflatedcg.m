function [x,d,a,beta,mu,p] = deflatedcg (k,l,A,b,x0,W,AW,tol)
% Space-inefficient implementation of Algorithm 1, DEFLATED-CG
% of Krylov Subspace Recycling for Fast Iterative Least-Squares in
% Machine Learning by de Roos and Hennig (2017)
% All indices shifted by one, because MATLAB
assert(size(W,2)==k, 'W needs to be n by k');
%TODO: need to change sizes!
d = zeros(l+1,1);
a = zeros(l+1,1);
beta = zeros(l+1,1);
mu = zeros(l+1,1);
p = zeros(l+1,1);
x = zeros(l+1,1);
r = zeros(l+1,1);

WTAW = transpose(W)*AW;
r0 = b - A*x0;
x(1) = x0 + (W*WTAW)\(transpose(W)*r0);
r(1) = b - A*x(1);
mu(1) = (WTAW)\(transpose(W)*A*r0);
p(1) = r(1) - W*mu(1);
j = 2;
while abs(res) > tol
    d(j-1) = transpose(p(j-1))*A*p(j-1);
    a(j-1) = transpose(r(j-1))*r(j-1)/d(j-1);
    x(j) = x(j-1)+a(j-1)*p(j-1);
    r(j) = r(j-1) - a(j-1)*A*p(j-1);
    beta(j-1) = transpose(r(j))*r(j)/(transpose(r(j-1))*r(j-1));
    mu(j) = (WTAW)\(transpose(W)*A*r(j));
    p(j) = beta(j-1)*p(j-1)+r(j)-W*mu(j);
    %if j <= l:
      %Store only in this case, but this one is space inefficient
end
x = x(j);
end

