function classifications = classify_SVM(X, weights, classes)
    [n, ~] = size(X);
    X_ones = [ones(n, 1) X];
    classifications = zeros(n, 1);
    
    for i=1:n
        if calc_distance_from_hyperplane(X_ones(i, :), weights) > 0
            classifications(i) = classes(2);
        else
            classifications(i) = classes(1);
        end
    end
end