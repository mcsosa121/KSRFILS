%{
    Implementation of the Conjugate Gradient Algorithm  
    
    Solves the problem Ax=b when A is symmetric and positive definite

    Input:
      A   - Problem matrix
      b   - Problem vector
      x0  - Problem guess. If unsure, just input x0 = 0
      mi  - Max number of iterations
      eps - Desired Accuracy
      M   - (Optional) Preconditioner. Recommended for large systems

    Output:
      x   - Solution to the problem
%}

function [x] = cg(A,b,x0,mi,eps,M)
    if nargin < 6
        % Non preconditioned CG
        x = nonpcg(A,b,x0,mi,eps);
    else
        % Preconditioned CG
        x = pcg(A,b,x0,mi,eps,M);
    end
end

% Non-Preconditioned Conjugate Gradient
function x = nonpcg(A,b,x0,mi,eps)
    % init
    i = 0;
    r = b - A*x0;
    d = r;
    x = x0;

    dtnew = transpose(r)*r;
    dt0 = dtnew;
    
    while (i < mi) && (dtnew > eps^2*dt0)
        disp('hi')
        q = A*d;
        alpha = dtnew / (transpose(d)*q);
        x = x + alpha*d;

        if mod(i,50) == 0
            r = b - A*x;
        else
            r = r - alpha*q;
        end

        dtold = dtnew;
        dtnew = transpose(r)*r;
        bta = dtnew / dtold;
        d = r + bta*d;
        i = i+1;
    end
end

% Preconditioned Conjugate Gradient
function x = pcg(A,b,x0,mi,eps,M)
    % init
    i = 0;
    r = b - A*x0;
    d = M \ r;
    x = x0;

    dtnew = transpose(r)*d;
    dt0 = dtnew;

    while (i < mi) && (dtnew > eps^2*dt0)
        q = A*d;
        alpha = dtnew / (transpose(d)*q);
        x = x + alpha*d;

        if mod(i,50) == 0
            r = b - A*x;
        else
            r = r - alpha*q;
        end

        s = M \ r;
        dtnew = transpose(r)*s;
        bta = dtnew / dtold;
        d = s + bta*d;
        i = i+1
    end
end