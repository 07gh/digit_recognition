function beta_vector = logit_one_vs_all_learn(X, y, start)
    addpath('..')

    classes = unique(y);
    [n, p] = size(X);
    k = numel(classes); % Get number of classes: k
    
    beta_vector = zeros(p+1, k); 
    classes_vector = zeros(k, 2);
    
    if nargin < 3
        % Initialize beta to zeros if user does not initialize
        start = zeros(p+1, 1);
    end
    
    for i=1:k
        % Run each class against all others
        y_two_classes =  zeros(n, 1);
        y_two_classes(y == classes(i)) = 1;
        y_two_classes(y ~= classes(i)) = 0;
        [beta, classes_i] = logistic_regression(X, y_two_classes, start);
        
        beta_vector(:, i) = beta;
        
        fprintf('Done with class %i out of %i\n', i, k)
    end
    
end