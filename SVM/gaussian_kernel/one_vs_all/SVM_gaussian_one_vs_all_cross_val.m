function [kernel_weights, X_kernel, gamma, classes] = SVM_gaussian_one_vs_all_cross_val(X_train, y_train, X_val, y_val)
addpath('..')

% Create Cs for cross-validation. Equidistant on a log scale
C = -6:12/40:6;
C = exp(C);
accuracy = zeros(41, 1);

% Run cross-validation
for i=1:41
    fprintf('Running cross-validation %i out of %i\n', i, 41)
    [kernel_weights, X_kernel, classes, gamma] = SVM_gaussian(X_train, y_train, C(i));
    classifications = classify_SVM_gaussian(X_val, kernel_weights, X_kernel, classes, gamma);
    accuracy(i) = sum(classifications == y_val) / numel(y_val);
end

% Get best value of lambda
[best_accuracy, best_i] = max(accuracy);
best_C = C(best_i);

% Get final model for class
[kernel_weights, X_kernel, classes, gamma] = SVM_gaussian(X_train, y_train, best_C);