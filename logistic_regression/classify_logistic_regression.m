function classifications = classify_logistic_regression(X, beta, classes, cutoff)
    [n, p] = size(X);

    if nargin < 4
        % Initialize cutoff value to .5 if user does not specify
        prob = .5;
    else
        prob = cutoff;
    end
    
    % Add ones column to X
    X_ones = [ones(n, 1) X];
    p = p + 1;
    
    classifications = zeros(n, 1);
    
    for i=1:n
        if calc_odds(X_ones(i, :), beta) >= prob
            classifications(i) = classes(2);
        else
            classifications(i) = classes(1);
        end
    end
end