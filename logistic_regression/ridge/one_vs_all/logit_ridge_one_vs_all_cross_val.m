function [beta, classes] = logit_ridge_one_vs_all_cross_val(X_train, y_train, X_val, y_val, start)
addpath('..')

% Create lambdas for cross-validation. Equidistant on a log scale
lambda = -6:12/40:6;
lambda = exp(lambda);
accuracy = zeros(41, 1);

% Run cross-validation
for i=1:41
    fprintf('Running cross-validation %i out of %i\n', i, 41)
    [beta, classes] = logistic_regression_ridge(X_train, y_train, lambda(i));
    classifications = classify_logistic_regression_ridge(X_val, beta, classes);
    accuracy(i) = sum(classifications == y_val) / numel(y_val);
end

% Get best value of lambda
[best_accuracy, best_i] = max(accuracy);
best_lambda = lambda(best_i);

% Get final model for class
[beta, classes] = logistic_regression_ridge(X_train, y_train, best_lambda, start);