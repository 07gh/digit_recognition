function classifications = logit_lasso_one_vs_all_classify(X, beta_vector, classes)
    k = numel(classes);
    [n, p] = size(X);
    
    prob_vector = zeros(n, k); % Difference in probability functions
    
    X = [ones(n, 1) X];
    
    for i=1:n
        for j=1:k
            prob_vector(i, j) = calc_odds(X(i, :), beta_vector(:, j));
        end
    end
    
    classifications = zeros(n, 1);
    for i=1:n
        [value, index] = max(prob_vector(i, :));
        classifications(i) = classes(index);
    end
end