% Tests SVM on digits 0 and 1.
% This achieves an accuracy of 99.36% on classifying 0s and 1s with a fraction of .9.

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
fraction = .9;

% Create validation set for cross-validation on C
X_val = X_train(1:int16(fraction*n_train), :);
y_val = y_train(1:int16(fraction*n_train), :);
n_val = numel(y_val);

X_train = X_train(int16(fraction*n_train)+1:n_train, :);
y_train = y_train(int16(fraction*n_train)+1:n_train, :);

lines_to_use = logical((y_test == 0) + (y_test == 1));
X_test = X_test(lines_to_use, :);
y_test = y_test(lines_to_use, :);

% Create Cs for cross-validation. Equidistant on a log scale
C = -6:12/40:6;
C = exp(C);
accuracy = zeros(41, 1);

disp('Running cross-validation. This may take time...')
for i=1:41
    [kernel_weights, X_kernel, classes, gamma] = SVM_gaussian(X_train, y_train, C(i));
    classifications = classify_SVM_gaussian(X_val, kernel_weights, X_kernel, classes, gamma);
    accuracy(i) = sum(classifications == y_val) / n_val;
    fprintf('Done with %i out of %i cross-validations\n', i, 41)
end

[best_accuracy, best_i] = max(accuracy);
best_C = C(best_i);
disp('Finished cross-validation')
fprintf('Best C value: %.2f\n', best_C)

disp('Learning...')
[kernel_weights, X_kernel, classes, gamma] = SVM_gaussian(X_train, y_train, best_C);
disp('Finished learning')

disp('Now testing...')
classifications = classify_SVM_gaussian(X_test, kernel_weights, X_kernel, classes, gamma);

fprintf('Fraction correct: %.4f\n', sum(classifications == y_test) / numel(y_test))