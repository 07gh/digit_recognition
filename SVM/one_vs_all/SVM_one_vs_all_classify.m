function classifications = SVM_one_vs_all_classify(X, weight_vector, classes)
    k = numel(classes);
    [n, p] = size(X);
    
    dist_vector = zeros(n, k); % Distance from hyperplane
    
    X = [ones(n, 1) X];
    
    for i=1:n
        for j=1:k
            dist_vector(i, j) = calc_distance_from_hyperplane(X(i, :), weight_vector(:, j));
        end
    end
    
    classifications = zeros(n, 1);
    for i=1:n
        [value, index] = max(dist_vector(i, :));
        classifications(i) = classes(index);
    end
end