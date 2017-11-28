%{
    Computes a matrix of approximate eigenvectors given a matrix A and desired
    number of eigenvectors.

    The algorithm works by using the Lanczos algorithm to produce a sequence
    of vectors which can then be used to transform A into a tri-diagonol
    and symmetric matrix. The eigendecomposition of this matrix give 
    a series of Ritz values which approximate the eigenvalues of A.
    Using these ritz values, corresponding eigenvectors can be found.

    Lanczos adapted from Prof Bindel's code from CS6210:
    https://github.com/dbindel/cs6210-f16/blob/49fce07c0427633bee13ef62fe9f1719ea22194e/lec/code/iter/lanczos.m
    Space saving equivalent of space_aeig.m, discarding old Q, since it is
    not needed.

    Input:
      A - Problem matrix
      k - Desired number of eigenvectors

    Output:
      V - Approximate eigenvectors
      e - Approximate eigenvalues
%}
function [V, e] = aeig(A, k)
  n = length(A);
  Q = zeros(n,k+1);   % Orthonormal basis
  alpha = zeros(k,1);
  beta  = zeros(k,1);
  b = randn(n,1);
  
  Qold = b/norm(b);
  for j = 1:k
    Qnew = A*Qold;
    alpha(j) = Qold'*Qnew;
    Qnew = Qnew-alpha(j)*Qold;
    if j > 1
      Qnew = Qnew-beta(j-1)*Qold2;
    end
    beta(j) = norm(Qnew);
    Qnew = Qnew/beta(j);
    Qold2 = Qold;
    Qold = Qnew;
  end
  
  T = diag(alpha) + diag(beta(1:k-1),1) + diag(beta(1:k-1),-1);
  [V,e] = eigs(T,k);
end