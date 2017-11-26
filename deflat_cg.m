% This program test the effect of higher order projection vectors
% and the deflation approach
%
% programmer: Kees Vuik
% e-mail    : c.vuik@math.tudelft.nl
% date      : 06-10-2005

clear
clf

n = 100; % dimension of matrix
m = 10;  % number of blocks

if mod(n,m) > 0
   disp('n is not a multiple of m')
   break
end

h = 1/n;

for deflation = 0: 3;
   
clear fout residu

for i = 1: n
   A(i,i) = 2;
   b(i,1) = sin(pi*h*i); % smooth right-hand-side vector
end

% b = rand(n,1); % random right-hand-side vector

for i = 1: n-1
   A(i+1,i) = -1;
   A(i,i+1) = -1;
end

A = sparse(A);

% xtrue = rand(n,1); % random solution vector
% b = A*xtrue;

% define the constant projection vectors

for i = 1: m
   Z(n,i) = 0;
   for j = 1:n/m
      Z(j+(i-1)*n/m,i) = 1;
   end
end

% define the linear projection vectors

if deflation > 0
   for i = 1: m
      Z(n,i+m) = 0;
      for j = 1:n/m
         Z(j+(i-1)*n/m,i+m) = j;
      end
   end
end

% define the quadratic projection vectors

if deflation > 1
   for i = 1: m
      Z(n,i+2*m) = 0;
      for j = 1:n/m
         Z(j+(i-1)*n/m,i+2*m) = j*j;
      end
   end
end

% define the cubic projection vectors

if deflation > 2
   for i = 1: m
      Z(n,i+3*m) = 0;
      for j = 1:n/m
         Z(j+(i-1)*n/m,i+3*m) = j*j*j;
      end
   end
end

Z  = sparse(Z);
AZ = sparse(A*Z);
E  = Z'*AZ;
P = sparse(eye(n)-AZ*inv(E)*Z');

% deflated cg

xtrue = A\b;
acc = 1d-6;
u = Z*inv(E)*Z'*b;
normxtrue = norm(xtrue);

i = 1;
x = 0*b;
r = P*b;
p = r;

residu(i) = norm(r);
fout(i)   = norm(xtrue-u)/normxtrue;

while  (i < 400) & (residu(i) > acc)
   i = i+1;
   PAp = P*(A*p);
   alpha = (r'*r)/(p'*PAp);
   x = x+alpha*p;
   r = r-alpha*PAp;
   beta = (r'*r)/(residu(i-1)^2);
   p = r+beta*p;
   residu(i) = norm(r);
   fout(i) = norm(xtrue-(u+P'*x))/normxtrue;
end

disp(['Number of iterations is: ' num2str(i)])

% yplot = residu; % plot the residu
yplot = fout;     % plot the relative error

if deflation == 0
   hplot(deflation+1) =  semilogy(yplot,'r');
end
if deflation == 1
   hplot(deflation+1) =  semilogy(yplot,'b');
end
if deflation == 2
   hplot(deflation+1) =  semilogy(yplot,'k');
end
if deflation == 3
   hplot(deflation+1) =  semilogy(yplot,'g');
end
hold on
end

set(gca,'FontSize', 15)
xlabel('Iteration number')
ylabel('(||x-x_k||_2)/||x||_2')

if deflation == 3
   legend(hplot,'constant','linear','quadratic','cubic')
end

