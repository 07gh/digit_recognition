function classifications = classify_LDA(X, means, C_inv, prob, classes)
    [k, dummy] = size(classes);
    [n, p] = size(X);
    disc = zeros(n, k); % The discriminant function for each class for each observation
    
    for i=1:n
        for j=1:k
            disc(i, j) = calc_LDA_discriminant(X(i, :), means, C_inv, prob, j);
        end
    end
    classifications = zeros(n, 1);
    for i=1:n
        [value, index] = max(disc(i, :));
        classifications(i) = classes(index);
    end
    
end