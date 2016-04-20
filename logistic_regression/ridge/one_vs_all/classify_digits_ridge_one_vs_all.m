% Tests logistic regression with ridge penalty on all digits.
% This achieves an accuracy of 89.19%.
addpath('../..')

data = csvread('../../../data/train.csv');
[n, p] = size(data);
y_train = data(:, 1);
X_train = data(:, 2:p);
n_train = numel(y_train);

data = csvread('../../../data/test.csv');
[n, p] = size(data);
y_test = data(:, 1);
X_test = data(:, 2:p);

% Create validation set for cross-validation on lambda
X_val = X_train(1:int16(n_train/3), :);
y_val = y_train(1:int16(n_train/3), :);
n_val = numel(y_val);

X_train = X_train(int16(n_train/3)+1:n_train, :);
y_train = y_train(int16(n_train/3)+1:n_train, :);

disp('Learning model...')
beta_vector = logit_ridge_one_vs_all_learn(X_train, y_train, X_val, y_val);
disp('Finished learning')

disp('Now testing...')
classifications = logit_ridge_one_vs_all_classify(X_test, beta_vector, unique(y_test));

fprintf('Fraction correct: %.4f\n', sum(classifications == y_test) / numel(y_test))