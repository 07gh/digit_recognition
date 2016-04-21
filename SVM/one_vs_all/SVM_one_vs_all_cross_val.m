function [weights, classes] = SVM_one_vs_all_cross_val(X_train, y_train, X_val, y_val)
addpath('..')

% Create Cs for cross-validation. Equidistant on a log scale
C = -6:12/40:6;
C = exp(C);
accuracy = zeros(41, 1);

% Run cross-validation
for i=1:41
    fprintf('Running cross-validation %i out of %i\n', i, 41)
    [weights, classes] = SVM(X_train, y_train, C(i));
    classifications = classify_SVM(X_val, weights, classes);
    accuracy(i) = sum(classifications == y_val) / numel(y_val);
end

% Get best value of lambda
[best_accuracy, best_i] = max(accuracy);
best_lambda = lambda(best_i);

% Get final model for class
[weights, classes] = logistic_regression_ridge(X_train, y_train, best_lambda, start);