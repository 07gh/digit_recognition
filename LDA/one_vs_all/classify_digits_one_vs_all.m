data = csvread('../../data/train.csv');
[n, p] = size(data);
y_train = data(:, 1);
X_train = data(:, 2:p);

data = csvread('../../data/test.csv');
[n, p] = size(data);
y_test = data(:, 1);
X_test = data(:, 2:p);

[mean_vector, C_inv_vector, prob_vector, classes] = LDA_one_vs_all_learn(X_train, y_train);

disp('Finished learning...')
disp('Now testing...')

classifications = LDA_one_vs_all_classify(X_test, mean_vector, C_inv_vector, prob_vector, classes);

disp('Percentage correct')
sum(classifications == y_test) / n