% Tests logistic regression on digits 0 and 1.
% Individually, this test does very well

data = csvread('../data/train.csv');
[n, p] = size(data);
y_train = data(:, 1);
X_train = data(:, 2:p);

data = csvread('../data/test.csv');
[n, p] = size(data);
y_test = data(:, 1);
X_test = data(:, 2:p);

lines_to_use = logical((y_train == 0) + (y_train == 1));
X_train = X_train(lines_to_use, :);
y_train = y_train(lines_to_use, :);

lines_to_use = logical((y_test == 0) + (y_test == 1));
X_test = X_test(lines_to_use, :);
y_test = y_test(lines_to_use, :);

[beta, classes] = logistic_regression(X_train, y_train);
disp('Finished learning...')
disp('Now testing...')
classifications = classify_logistic_regression(X_test, beta, classes);

fprintf('Fraction correct: %.4f\n', sum(classifications == y_test) / numel(y_test))


