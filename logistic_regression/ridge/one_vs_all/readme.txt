Logistic Regression with Ridge Penalty with one vs. all multiclassification

One vs. all achieves 88.29% accuracy.
classify_digits_ridge_one_vs_all.m runs the classification. This classification technique runs cross-validation on lambda.

Learn the model on logit_ridge_one_vs_all_learn.m. Automatically adds a column of ones
Inputs:
	X_train -- Input matrix of size n x p. Training set
		n -- Number of observations.
		p -- Number of predictors.
		X(i, j) =  predictor value j of observation i
	y_train -- Class corresponding to X_train. Size n x 1. Training set
		The value in row i of y is the class of observation i in X_train
	X_val -- Input matrix of size n x p. Validation set
	y_val -- Class corresponding to X. Size n x 1. Validation set
	start -- Optional parameter. You may specify what value of beta (vector of size p+1) to start the gradient descent algorithm. Default is a vector of zeros.

Outputs:
	beta_vector -- The vector of betas of each predictor for each class
		beta_vector(i, j) = coefficient of predictor i for class j

Once you've learned the model, test new observations using logit_one_vs_all_classify.m
Inputs:
	X -- Data to be classified. Same format as X in logit_ridge_one_vs_all_learn.m (Do not include column of ones)
	beta_vector -- Output from logit_ridge_one_vs_all_learn.m
	classes -- Unique classes

Outputs:
	classifications -- classes predicted for X
		classifications(i) is the class predicted for observation i in X

logit_ridge_one_vs_all_cross_val.m. Helper function for logit_ridge_one_vs_all_learn.m. Runs cross validation on lambda.
Inputs:
	X_train -- Same as in logit_ridge_one_vs_all_learn.m
	y_train -- Same as in logit_ridge_one_vs_all_learn.mtrain
	X_val -- Same as in logit_ridge_one_vs_all_learn.m
	y_val -- Same as in logit_ridge_one_vs_all_learn.m
	start -- Same as in logit_ridge_one_vs_all_learn.m

Outputs:
	beta -- Beta for the current class in one vs. all multi-classification
	classes -- Unique classes