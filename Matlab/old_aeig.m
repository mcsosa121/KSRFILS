%{
    Computes a matrix of approximate eigenvectors given a matrix A and desired
    number of eigenvectors.

    The algorithm works by using the Lanczos algorithm to produce a sequence
    of vectors which can then be used to transform A into a tri-diagonol
    and symmetric matrix. The eigendecomposition of this matrix give 
    a series of Ritz values which approximate the eigenvalues of A.
    Using these ritz values, corresponding eigenvectors can be found.

    Input:
      A - Problem matrix
      k - Desired number of eigenvectors
%}
function [W, e] = old_aeig(A, k)
    % need to implement
    n = size(A,1);
    v = randn(n,1);
    v = v/norm(v); %don't need v to be that random
    T = zeros(n,n);
    
    %initial iteration step
    wp = A*v;
    a = wp'*v;
    w = wp - a*v;
    T(1,1) = a;
    
    for j = 2:k
        b = norm(w);
        vprev = v;
        if b ~= 0
            v = w/b;
        else
            v = randn(n,1);
            v = v/norm(v);
        end
        wp = A*v;
        a = wp'*v;
        w = wp - a*v - b*vprev;
        T(j,j) = a;
        T(j-1,j) = b;
        T(j,j-1) = b;
    end
    [W,e] = eigs(T,k);
end