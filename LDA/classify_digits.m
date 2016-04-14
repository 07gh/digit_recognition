data = csvread('../data/train.csv');
[n, p] = size(data);
y_train = data(:, 1);
X_train = data(:, 2:p);

data = csvread('../data/test.csv');
[n, p] = size(data);
y_test = data(:, 1);
X_test = data(:, 2:p);

[means, C_inv, prob, classes] = LDA(X_train, y_train);
disp('Finished learning...')
disp('Now testing...')
classifications = classify_LDA(X_test, means, C_inv, prob, classes);

disp('Percentage correct')
sum(classifications == y_test) / n
