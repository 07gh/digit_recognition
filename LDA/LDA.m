function [means, C_inv, prob, classes] = LDA(X, y)
    classes = unique(y);
    [k, dummy] = size(classes); % Get number of classes: k
    [n, p] = size(X); % Get dimensions of X: n, p
    
    means = zeros(k, p); % Stores the mean of each predictor for each class
    mean_overall = zeros(1, p); % Stores the overall mean of each predictor
    for j=1:p
        for i=1:k
            means(i, j) = mean( X(y==classes(i), j) );
        end
        mean_overall(j) = mean( X(:, j) );
    end
    
    X_minus_mean = zeros(n, p); % X minus overall mean
    for j=1:p
        X_minus_mean(:, j) = X(:, j) - mean_overall(j);
    end
    
    cov_matrices = zeros(k, p, p); % Covariance matrices for each class
    counts = zeros(k, 1); % Number of observations in each class
    for i=1:k
       class_i = X_minus_mean(y==classes(i), :);
       counts(i) = sum(y==classes(i));
       cov_matrices(i, :, :) =  class_i' * class_i / counts(i);
    end
    
    C = zeros(p, p); % Group covariance matrix
    for i=1:p
        for j=1:p
            C(i, j) = sum(counts .* cov_matrices(:, i, j)) / n;
        end
    end
    
    C_inv = inv(C); % Inverse of group covariance matrix
    
    prob = counts / n; % Prior probabilities of classes
end