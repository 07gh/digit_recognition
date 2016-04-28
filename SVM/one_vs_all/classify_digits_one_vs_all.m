% Tests SVM on all digits.
% This achieves an accuracy of 89.12% with a fraction of 70%.
addpath('..')

data = csvread('../../data/train.csv');
[n, p] = size(data);
y_train = data(:, 1);
X_train = data(:, 2:p);
n_train = numel(y_train);

data = csvread('../../data/test.csv');
[n, p] = size(data);
y_test = data(:, 1);
X_test = data(:, 2:p);

fraction = .7;

% Create validation set for cross-validation on C
X_val = X_train(1:int16(fraction*n_train), :);
y_val = y_train(1:int16(fraction*n_train), :);
n_val = numel(y_val);

X_train = X_train(int16(fraction*n_train)+1:n_train, :);
y_train = y_train(int16(fraction*n_train)+1:n_train, :);

disp('Learning model...')
weight_vector = SVM_one_vs_all_learn(X_train, y_train, X_val, y_val);
disp('Finished learning')

disp('Now testing...')
classifications = SVM_one_vs_all_classify(X_test, weight_vector, unique(y_test));

fprintf('Fraction correct: %.4f\n', sum(classifications == y_test) / numel(y_test))