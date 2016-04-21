% Tests logistic regression with lasso penalty on digits 0 and 1.
% This achieves an accuracy of 98.88% on classifying 0s and 1s.

data = csvread('../../data/train.csv');
[n, p] = size(data);
y_train = data(:, 1);
X_train = data(:, 2:p);

data = csvread('../../data/test.csv');
[n, p] = size(data);
y_test = data(:, 1);
X_test = data(:, 2:p);

lines_to_use = logical((y_train == 0) + (y_train == 1));
X_train = X_train(lines_to_use, :);
y_train = y_train(lines_to_use, :);
n_train = numel(y_train);

% Create validation set for cross-validation on lambda
X_val = X_train(1:n_train/3, :);
y_val = y_train(1:n_train/3, :);
n_val = numel(y_val);

X_train = X_train(n_train/3+1:n_train, :);
y_train = y_train(n_train/3+1:n_train, :);

lines_to_use = logical((y_test == 0) + (y_test == 1));
X_test = X_test(lines_to_use, :);
y_test = y_test(lines_to_use, :);

% Create lambdas for cross-validation. Equidistant on a log scale
lambda = -6:12/40:6;
lambda = exp(lambda);
accuracy = zeros(41, 1);

disp('Running cross-validation. This may take time...')
for i=1:41
    [beta, classes] = logistic_regression_lasso(X_train, y_train, lambda(i));
    classifications = classify_logistic_regression_lasso(X_val, beta, classes);
    accuracy(i) = sum(classifications == y_val) / n_val;
    fprintf('Done with %i out of %i cross-validations\n', i, 41)
end

[best_accuracy, best_i] = max(accuracy);
best_lambda = lambda(best_i);
disp('Finished cross-validation')
fprintf('Best lambda value: %.2f\n', best_lambda)

disp('Learning...')
[beta, classes] = logistic_regression_lasso(X_train, y_train, best_lambda);
disp('Finished learning')

disp('Now testing...')
classifications = classify_logistic_regression_lasso(X_test, beta, classes);

fprintf('Fraction correct: %.4f\n', sum(classifications == y_test) / numel(y_test))