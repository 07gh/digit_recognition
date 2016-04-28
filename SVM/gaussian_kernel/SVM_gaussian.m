function [kernel_weights, X_ones, classes, gamma] = SVM_gaussian(X, y, C)
    % Runs an SVM, adding a column of ones
    classes = unique(y);
    if numel(classes) ~= 2
        error('SVM requires only 2 classes.')
    end
    
    % Convert classes to one and negative one
    [n, ~] = size(y);
    y_one_neg_one = zeros(n, 1);
    y_one_neg_one(y == classes(2)) = 1;
    y_one_neg_one(y == classes(1)) = -1;
    %Y = diag(y_one_neg_one); % Set diagonal matrix  Y
    
    X_ones = [ones(n, 1) X];
    
    % Set H matrix
    %X1 = X_ones(y_one_neg_one == 1, :);
    %z = mean(X1); % Row vector containing means of each column for class 1
    gamma = -.5/var(X_ones(:)); % Get variance of entire matrix
    
    H = zeros(n);
    for i=1:n
        for j=1:n
            H(i, j) = y_one_neg_one(i) * y_one_neg_one(j) * ...
                kernel(X_ones(i, :), X_ones(j, :), gamma);
        end
    end
    
    lb = zeros(n, 1); % Set lower bound for lambda
    ub = ones(n, 1) * C; % Set upper bound for lambda
    
    f = ones(n, 1) * -1; % Set ones vector 
    
    warning('off', 'optim:quadprog:HessianNotSym')
    options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','off');%,'MaxIter',2);
    lambda = quadprog(H, f, [], [], [], [], lb, ub, [], options);
    
    kernel_weights = y_one_neg_one .* lambda;
end