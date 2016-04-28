function classifications = classify_SVM_gaussian(X, kernel_weights, X_kernel, classes, gamma)
    [n, ~] = size(X);
    X_ones = [ones(n, 1) X];
    classifications = zeros(n, 1);
    
    for i=1:n
        if calc_distance_from_hyperplane_gaussian(X_ones(i, :), kernel_weights, X_kernel, gamma) > 0
            classifications(i) = classes(2);
        else
            classifications(i) = classes(1);
        end
    end
end