%{
  Space efficient implementation of Deflated-CG
  
  Solves the problem Ax=b where A is Symmetric and Positive Definite

  The point of deflation is to "deflate" or "shrink" the solution to a simpler
  solution space so that convergence is sped up. 

  An example of the deflated CG process can be seen below
    Problem: Ax=b 
    1. CG(A,b) --> A^{-1}b [Initial solution by normal conjugate gradient]
    2. A tri-diagnol low rank approximation of A gives us a matrix of approximate eigenvectors W
    3. Projection P_W is used to a preconditioner 
    4. Deflated CG takes over
n    5. Repeat steps 1-4 with results from Deflated CG
  Refer to 
    A Deflated Version of the Conjugate Gradient Algorithm
    Y. Saad, M. Yeung, J. Erhel, and F. Guyomarc'h
    SIAM Journal on Scientific Computing 2000 21:5, 1909-1926 
  For more details


  Input:
    k   - Number of approximate eigenvectors
    A   - Problem matrix
    b   - Problem vector
    W   - Matrix of approximate eigenvectors from a low rank approx
    eps - Desired tolerance
    x0  - Initial guess. If unsure put [0,0,...,0]. CG is guarenteed to converge
            provided A is SPD
    AW  - (Optional) Can be provided if A and W are able to be cheaply multiplied
            beforehand. If not provided supply [] as the input.
    M   - (Optional) Optional Preconditioner. If not provided supply [] as input or nothing at all.
    mi  - (Optional) Max iterations before the algorithm terminates. If not supplied defaulted to 500.

  Output:
    x   - Solution to the problem
%}

function [x] = defcg(k,A,b,x0,W,eps,AW,M,mi)
    % Make sure that W \in R^{n,k}
    assert(size(W,2)==k, 'W needs to be n by k');
    
    % Checking if Mi is provided
    if ~exist('mi', 'var') || isempty(mi)
      mi = 500;
    end

    % Checking if AW is provided
    if ~exist('AW', 'var') || isempty(AW)
      AW = A*W;
    end

    % Non-Preconditioned Deflated-CG
    if ~exist('M','var') || isempty(M)
      x = nonpcdcg(A,b,x0,W,eps,AW,mi);
    else
      % Preconditioned Deflated-CG
      x = pcdcg(A,b,x0,W,eps,AW,M,mi);
    end
end


% Non-Preconditioned Deflated-CG
function [x] = nonpcdcg(A,b,x0,W,eps,AW,mi)
    % Init
    WTAW = transpose(W)*AW;
    i = 0;
    r = b - A*x0;

    % Choose x such that W^T*r = 0 where r = b - Ax
    Wr = transpose(W)*r;
    Wr = WTAW \ Wr;
    x = x0 + W*Wr; 

    % Calculate updated residual and deltas
    r = b - A*x;
    % Solve W^T*A*W*mu = W^T*A*r for mu
    mu = WTAW \ (transpose(W)*A*r);
    dtnew = transpose(r)*r;
    dt0 = dtnew;
    p = r - W*mu;    
    
    while (i < mi) && (dtnew > eps^2*dt0)
        q = A*p;
        % periodically calculate residual explicitely to avoid numerical
        % error from recurrence;
        alpha = dtnew / (transpose(p)*q);
        x = x + alpha*p;

        % periodically calculate residual explicitely to avoid numerical
        % error from recurrence
        if mod(i,50) == 0
            r = b - A*x;
        else
            r = r - alpha*q;
        end

        dtold = dtnew;
        dtnew = transpose(r)*r;
        bta = dtnew / dtold;

        % Possibly eliminate this somehow?
        mu = WTAW \ (transpose(W)*A*r)
        p = bta*p + r - W*mu;
        
        i = i+1;
    end
end


% Preconditioned Deflated-CG
function [x] = pcdcg(A,b,x0,W,eps,AW,M,mi)
    % Init
    WTAW = transpose(W)*AW;
    i = 0;
    r = b - A*x0;

    % Choose x such that W^T*r = 0 where r = b - Ax
    Wr = transpose(W)*r;
    Wr = WTAW \ Wr;
    x = x0 + W*Wr;

    % Calculate update residual and z
    r = b-A*x;
    z = M \ r;
    % Solve W^T*A*W*mu = W^T*A*z for mu
    mu = WTAW \ (transpose(W)*A*z);
    dtnew = transpose(r)*z;
    dt0 = dtnew;
    p = z - W*mu;

    while (i < mi) && (dtnew > eps^2*dt0)
        q = A*p;
        alpha = dtnew / (transpose(p)*q);
        x = x + alpha*p;

        % periodically calculate residual explicitely to avoid numerical
        % error from recurrence 
        if mod(i,50) == 0
            r = b - A*x;
        else
            r = r - alpha*q;
        end

        z = M \ r;
        dtold = dtnew;
        dtnew = transpose(r)*z;
        bta = dtnew / dtold;

        mu = WTAW \ (transpose(W)*A*z);
        p = bta*p + z - W*mu;
        i = i+1;
    end
end