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
function [W] = aeig(A)
    % need to implement
    W = A;
end