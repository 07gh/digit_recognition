function [beta, classes] = logistic_regression_ridge(X, y, lambda, start)
    % Note a column of ones is added to X, 
    [n, p] = size(X);
    classes = unique(y);
    
    % Add ones column to X
    X_ones = [ones(n, 1) X];
    p = p + 1;
    
    % Set gradient descent parameters
    alpha = .01;
    k = 0;
    max_iter = 100;
    tol = .1;
    if nargin < 4
        % Initialize beta to zeros if user does not initialize
        beta = zeros(p, 1);
    else
        beta = start;
    end
    old_beta = ones(p, 1);
    
    % Convert classes to 1 and -1
    y_one_neg_one = zeros(n, 1);
    y_one_neg_one(y == classes(1)) = -1;
    y_one_neg_one(y == classes(2)) = 1;
    
    while norm(beta - old_beta) >= tol && k <= max_iter
        grad = calc_gradient_ridge(beta, lambda, X_ones, y_one_neg_one);
        d = -grad;
        old_beta = beta;
        beta = beta + alpha*d;
        k = k + 1;
    end
end