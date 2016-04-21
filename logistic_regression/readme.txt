Logistic Regression

classify_digits.m runs the classification on 2 classes. For > 2 classes, use the one_vs_all classification.

NB: Do not include a column of ones in X. The functions add it in automatically.

Learn the model on logistic_regression.m
Inputs:
	X -- Input matrix of size n x p
		n -- Number of observations.
		p -- Number of predictors.
		X(i, j) =  predictor value j of observation i
	y -- Class corresponding to X. Size n x 1
		The value in row i of y is the class of observation i in X
	start -- Optional parameter. You may specify what value of beta (vector of size p+1) to start the gradient descent algorithm. Default is a vector of zeros.

Outputs:
	beta -- Coefficients for logistic regression, including intercept in first spot.
	classes -- The 2 classes.
		Logistic regression uses classes 1 and -1 for its algorithms, but user can use any two classes without any problems.

Once you've learned the model, test new observations using classify_logistic_regression.m
Inputs:
	X -- Data to be classified. Same format as X in logistic_regression.m (Do not add column of ones)
	beta -- Output from logistic_regression.m
	classes -- Output from logistic_regression.m
	cutoff -- Optional parameter. Decision boundary for choosing classes. 
		Can be value between 0 and 1. Default is .5.

Outputs:
	classifications -- classes predicted for X
		classifications(i) is the class predicted for observation i in X

calc_odds.m is a helper function for classify_logistic_regression.m
It calculates the odds for the given class for the given observation
Inputs:
	x -- Observation to calculate odds for. One row in X from classify_logistic_regression.m
	beta -- Same as logistic_regression.m

Outputs:
	odds -- Odds of being class 1. Value between 0 and 1.

calc_gradient.m is a helper function for logistic_regression.m
It calculates the gradient for use in the gradient descent method
Inputs:
	beta -- Current iteration of beta in the gradient descent method
	X -- Same as in logistic_regression.m with the column of ones added.
	y -- Same as in logistic_regression.m with classes 1 and -1.

Outputs:
	grad -- Value of gradient for given values of beta, X, and y
