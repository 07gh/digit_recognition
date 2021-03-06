Logistic Regression with one vs. all multiclassification

One vs. all achieves 85.55% accuracy.
classify_digits_one_vs_all.m runs the classification.

Learn the model on logit_one_vs_all_learn.m. Automatically adds a column of ones
Inputs:
	X -- Input matrix of size n x p
		n -- Number of observations.
		p -- Number of predictors.
		X(i, j) =  predictor value j of observation i
	y -- Class corresponding to X. Size n x 1
		The value in row i of y is the class of observation i in X
	start -- Optional parameter. You may specify what value of beta (vector of size p+1) to start the gradient descent algorithm. Default is a vector of zeros.

Outputs:
	beta_vector -- The vector of betas of each predictor for each class
		beta_vector(i, j) = coefficient of predictor i for class j

Once you've learned the model, test new observations using logit_one_vs_all_classify.m
Inputs:
	X -- Data to be classified. Same format as X in logit_one_vs_all_learn.m (Do not include column of ones)
	beta_vector -- Output from logit_one_vs_all_learn.m
	classes -- Unique classes

Outputs:
	classifications -- classes predicted for X
		classifications(i) is the class predicted for observation i in X