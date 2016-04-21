function [weights, classes] = SVM(X, y, C)
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
    Y = diag(y_one_neg_one); % Set diagonal matrix  Y
    
    X_ones = [ones(n, 1) X];
    
    % Set H matrix
    H = Y'*X_ones*X_ones'*Y;
    lb = zeros(n, 1); % Set lower bound for lambda
    ub = ones(n, 1) * C; % Set upper bound for lambda
    
    f = ones(n, 1) * -1; % Set ones vector 
    
    options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','off');
    lambda = quadprog(H, f, [], [], [], [], lb, ub, [], options);
    
    weights = X_ones'*Y*lambda;
end