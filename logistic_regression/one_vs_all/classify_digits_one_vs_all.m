data = csvread('../../data/train.csv');
[n, p] = size(data);
y_train = data(:, 1);
X_train = data(:, 2:p);

data = csvread('../../data/test.csv');
[n, p] = size(data);
y_test = data(:, 1);
X_test = data(:, 2:p);

classes = unique(y_train);
k = numel(classes);

fprintf('Running %i logistic models. This may take time...\n', k)

beta_vector = logit_one_vs_all_learn(X_train, y_train);

disp('Finished learning...')
disp('Now testing...')

classes = unique(y_test);
classifications = logit_one_vs_all_classify(X_test, beta_vector, classes);

disp('Percentage correct')
sum(classifications == y_test) / n