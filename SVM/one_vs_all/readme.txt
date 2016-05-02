Support Vector Machine (SVM) with one vs. all multiclassification
-- No kernel (i.e. linear kernel)

One vs. all achieves 89.12% accuracy.
Fraction of 70% (Meaning 70% of the training dataset is used for the validation set, while 30% of the dataset is used for the training set)
classify_digits_one_vs_all.m runs the classification.

Learn the model on SVM_one_vs_all_learn.m. Automatically adds a column of ones
Inputs:
	X_train -- Input matrix of size n x p
		n -- Number of observations.
		p -- Number of predictors.
		X(i, j) =  predictor value j of observation i
	y_train -- Class corresponding to X. Size n x 1
		The value in row i of y is the class of observation i in X
	X_val -- Input matrix of size n x p. Validation set
	y_val -- Class corresponding to X. Size n x 1. Validation set

Outputs:
	weight_vector -- The vector of weights of each predictor for each class
		weight_vector(i, j) = coefficient of predictor i for class j

Once you've learned the model, test new observations using SVM_one_vs_all_classify.m
Inputs:
	X -- Data to be classified. Same format as X in SVM_one_vs_all_learn.m (Do not include column of ones)
	weight_vector -- Output from SVM_one_vs_all_learn.m
	classes -- Unique classes

Outputs:
	classifications -- classes predicted for X
		classifications(i) is the class predicted for observation i in X

SVM_one_vs_all_cross_val.m. Helper function for SVM_one_vs_all_learn.m. Runs cross validation on C.
Inputs:
	X_train -- Same as in SVM_one_vs_all_learn.m
	y_train -- Same as in SVM_one_vs_all_learn.m
	X_val -- Same as in SVM_one_vs_all_learn.m
	y_val -- Same as in SVM_one_vs_all_learn.m

Outputs:
	weights -- Weights for given class
	classes -- For recall in classification