Support Vector Machine (SVM)

classify_digits.m runs the classification on 2 classes. For > 2 classes, use the one_vs_all classification. Runs cross-validation on C.

NB: Do not include a column of ones in X. The functions add it in automatically.

Learn the model on SVM.m
Inputs:
	X -- Input matrix of size n x p
		n -- Number of observations.
		p -- Number of predictors.
		X(i, j) =  predictor value j of observation i
	y -- Class corresponding to X. Size n x 1
		The value in row i of y is the class of observation i in X
	C -- Error penalty.

Outputs:
	weights -- Coefficients for SVM, including intercept in first spot.
	classes -- The 2 classes.
		SVM uses classes 1 and -1 for its algorithms, but user can use any two classes without any problems because the functions automatically convert to classes 1 and -1.

Once you've learned the model, test new observations using classify_SVM.m
Inputs:
	X -- Data to be classified. Same format as X in SVM.m (Do not add column of ones)
	weights -- Output from SVM.m
	classes -- Output from SVM.m

Outputs:
	classifications -- classes predicted for X
		classifications(i) is the class predicted for observation i in X

calc_distance_from_hyperplane.m is a helper function for classify_SVM.m
It calculates the distance from the hyperplane for the given observation
Inputs:
	x -- Observation to calculate distance from hyperplane. One row in X from classify_SVM.m
	weights -- Same as SVM.m

Outputs:
	dist -- Distance from hyperplane, including direction (positive is 'above' hyperplane, while negative is 'below' hyperplane)