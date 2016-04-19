function beta_vector = logit_ridge_one_vs_all_learn(X_train, y_train, X_val, y_val, start)
    addpath('..')

    classes = unique(y_train);
    [n_train, p] = size(X_train);
    [n_val, p] = size(X_val);
    k = numel(classes); % Get number of classes: k
    
    beta_vector = zeros(p+1, k); 
    
    if nargin < 5
        % Initialize beta to zeros if user does not initialize
        start = zeros(p+1, 1);
    end
    
    for i=1:k
        fprintf('Running class %i out of %i...\n', i, k)
        % Run each class against all others
        y_train_two_classes =  zeros(n_train, 1);
        y_train_two_classes(y_train == classes(i)) = 1;
        y_train_two_classes(y_train ~= classes(i)) = 0;
        y_val_two_classes =  zeros(n_val, 1);
        y_val_two_classes(y_val == classes(i)) = 1;
        y_val_two_classes(y_val ~= classes(i)) = 0;
        [beta, classes_i] = logit_ridge_one_vs_all_cross_val(X_train, y_train_two_classes, X_val, y_val_two_classes, start);
        
        beta_vector(:, i) = beta;
    end
end