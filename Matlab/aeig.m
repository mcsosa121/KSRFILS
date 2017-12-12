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
    
    V obeys r = (A ? ?I)v ? K_k(A,b), for all v in V.

    Input:
      A - Problem matrix
      k - Desired number of eigenvectors
      b - Starting vector (should be random, e.g. randn(n,1))

    Output:
      V - Approximate eigenvectors
      e - Approximate eigenvalues
%}
function [V, e] = aeig(A, k, b)
  n = length(A);
  Q = zeros(n,k+1);   % Orthonormal basis
  alpha = zeros(k+1,1);
  beta  = zeros(k,1);  
  
  Q(:,1) = b/norm(b);
  for j = 1:k
    Q(:,j+1) = A*Q(:,j);
    alpha(j) = Q(:,j)'*Q(:,j+1);
    Q(:,j+1) = Q(:,j+1)-alpha(j)*Q(:,j);
    if j > 1
      Q(:,j+1) = Q(:,j+1)-beta(j-1)*Q(:,j-1);
    end
    beta(j) = norm(Q(:,j+1));
    Q(:,j+1) = Q(:,j+1)/beta(j);
    
    sum = 0;
    for i = 1:j-1
        sum = sum+(Q(:,j+1)'*Q(:,i))*Q(:,i);
    end
    Q(:,j+1) = Q(:,j+1) - sum;
  end
  
  T = diag(alpha) + diag(beta,1) + diag(beta,-1);
  [W,e] = eigs(T,k);  
  
  V = Q*W;
end