% Tests SVM with Gaussian kernel on all digits.
% This achieves an accuracy of 91.93% with a validation fraction of .15 and a training fraction of .3.
addpath('..')

data = csvread('../../../data/train.csv');
[n, p] = size(data);
y_train = data(:, 1);
X_train = data(:, 2:p);
n_train = numel(y_train);

data = csvread('../../../data/test.csv');
[n, p] = size(data);
y_test = data(:, 1);
X_test = data(:, 2:p);

val_fraction = .15;
train_fraction = .3;

% Create validation set for cross-validation on C
X_val = X_train(1:int16(val_fraction*n_train), :);
y_val = y_train(1:int16(val_fraction*n_train), :);
n_val = numel(y_val);

X_train = X_train(int16(val_fraction*n_train)+1:int16((val_fraction+train_fraction)*n_train), :);
y_train = y_train(int16(val_fraction*n_train)+1:int16((val_fraction+train_fraction)*n_train), :);

disp('Learning model...')
[kernel_weight_vector, X_kernel_vector, gamma] = SVM_gaussian_one_vs_all_learn(X_train, y_train, X_val, y_val);
disp('Finished learning')

disp('Now testing...')
classifications = SVM_gaussian_one_vs_all_classify(X_test, kernel_weight_vector, X_kernel_vector, gamma, unique(y_test));

fprintf('Fraction correct: %.4f\n', sum(classifications == y_test) / numel(y_test))