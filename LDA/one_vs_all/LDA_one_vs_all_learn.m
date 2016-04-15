function [mean_vector, C_inv_vector, prob_vector, classes] = LDA_one_vs_all_learn(X, y)
    addpath('..')

    classes = unique(y);
    [n, p] = size(X);
    k = numel(classes); % Get number of classes: k
    
    mean_vector = zeros(2, p, k);
    C_inv_vector = zeros(p, p, k);
    prob_vector = zeros(2, k);
    
    for i=1:k
        % Run each class against all others
        y_two_classes =  zeros(n, 1);
        y_two_classes(y == classes(i)) = 1;
        y_two_classes(y ~= classes(i)) = 0;
        [means, C_inv, prob, classes_i] = LDA(X, y_two_classes);
        mean_vector(:, :, i) = means;
        C_inv_vector(:, :, i) = C_inv;
        prob_vector(:, i) = prob;
    end
end