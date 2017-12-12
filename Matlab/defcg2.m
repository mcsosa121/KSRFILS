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
        % initial iteration
        [x,mui] = nonpcdcg(A,b,x0,W,eps,AW,mi);
        disp(x);
    else
      % Preconditioned Deflated-CG
      [x,mui] = pcdcg(A,b,x0,W,eps,AW,M,mi);
    end
end


% Non-Preconditioned Deflated-CG
function [x,mu] = nonpcdcg(l,A,b,x0,W,eps,AW,mi,mu)
    % Init
    WTAW = transpose(W)*AW;
    i = 0;
    r0 = b - A*x0;
    
    %storage
    ds = zeros(l,1);
    as = zeros(l,1);
    bs = zeros(l,1);
    m = zeros(l,1);
    p = zeros(l,1);
    
    
    % Choose x such that W^T*r = 0 where r = b - Ax
    Wr = transpose(W)*r0;
    Wr = WTAW \ Wr;
    x = x0 + W*Wr; 

    % Calculate updated residual and deltas
    r = b - A*x;

    % Solve W^T*A*W*mu = W^T*A*r for mu if initial iteration
    if nargin < 8
        m(0) = WTAW \ (transpose(W)*A*r);
    else
        m(0) = mu;
    end
    p(0) = r - W*mu;
    
    dtnew = transpose(p(0))*A*p(0);
    dt0 = dtnew;
     
    
    while (i < mi) && (dtnew > eps^2*dt0)
        d(j-1) = p(j-1)*A*p(0);
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
        mu = WTAW \ (transpose(W)*A*r);
        p = bta*p + r - W*mu;
        
        i = i+1;
        if j < l
           ds(j-1) = dtnew;
           as(j-1) = alpha;
           bs(j-1) = bta;
           m(j) = mu;
           p(j) = p;
        end    
        j = j+1
    end
end


% Preconditioned Deflated-CG
function [x,mu] = pcdcg(A,b,x0,W,eps,AW,M,mi,mu)
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
    % Solve W^T*A*W*mu = W^T*A*z for mu if intial iteration
    if nargin < 9
        mu = WTAW \ (transpose(W)*A*z);
    end
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