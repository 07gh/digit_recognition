function [kernel_weight_vector, X_kernel_vector, gamma] = SVM_gaussian_one_vs_all_learn(X_train, y_train, X_val, y_val)
    addpath('..')

    classes = unique(y_train);
    [n_train, p] = size(X_train);
    [n_val, p] = size(X_val);
    k = numel(classes); % Get number of classes: k
    
    kernel_weight_vector = zeros(n_train, k);
    X_kernel_vector = zeros(n_train, p+1, k);
    
    for i=1:k
        fprintf('Running class %i out of %i...\n', i, k)
        % Run each class against all others
        y_train_two_classes =  zeros(n_train, 1);
        y_train_two_classes(y_train == classes(i)) = 1;
        y_train_two_classes(y_train ~= classes(i)) = 0;
        y_val_two_classes =  zeros(n_val, 1);
        y_val_two_classes(y_val == classes(i)) = 1;
        y_val_two_classes(y_val ~= classes(i)) = 0;
        [kernel_weights, X_kernel, gamma, ~] = SVM_gaussian_one_vs_all_cross_val(X_train, y_train_two_classes, X_val, y_val_two_classes);
        
        kernel_weight_vector(:, i) = kernel_weights;
        X_kernel_vector(:, :, i) = X_kernel;
    end
end