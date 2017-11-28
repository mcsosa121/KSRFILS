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
    
    Input:
      A - Problem matrix
      k - Desired number of eigenvectors

    Output:
      V - Approximate eigenvectors
      e - Approximate eigenvalues
%}
function [V, e] = aeig(A, k)
    j = 2*k;
    n = size(A,1);
    v = randn(n,1);
    v = v/norm(v);
    Q = zeros(n,j+1);
    alpha = zeros(j,1);
    beta = zeros(j,1);
    
    Q(:,1) = v/norm(v);
    for j = 1:j
        Q(:,j+1) = A*Q(:,j);
        alpha(j) = Q(:,j)'*Q(:,j+1);
        Q(:,j+1) = Q(:,j+1)-alpha(j)*Q(:,j);
        if j > 1
          Q(:,j+1) = Q(:,j+1)-beta(j-1)*Q(:,j-1);
        end
        beta(j) = norm(Q(:,j+1));
        Q(:,j+1) = Q(:,j+1)/beta(j);
    end
    T = diag(alpha) + diag(beta(1:j-1),1) + diag(beta(1:j-1),-1);
    [V,e] = eigs(T, k);
end