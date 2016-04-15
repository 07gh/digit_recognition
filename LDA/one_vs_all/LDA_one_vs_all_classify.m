function classifications = LDA_one_vs_all_classify(X, mean_vector, C_inv_vector, prob_vector, classes)
    k = numel(classes);
    [n, p] = size(X);
    
    disc_diffs = zeros(n, k); % Difference in discriminant functions
    
    for i=1:n
        for j=1:k
            disc_diffs(i, j) = calc_LDA_discriminant(X(i, :), mean_vector(:, :, j), C_inv_vector(:, :, j), prob_vector(:, j), 2) - ...
                  calc_LDA_discriminant(X(i, :), mean_vector(:, :, j), C_inv_vector(:, :, j), prob_vector(:, j), 1);
        end
    end
    
    classifications = zeros(n, 1);
    for i=1:n
        [value, index] = max(disc_diffs(i, :));
        classifications(i) = classes(index);
    end
end