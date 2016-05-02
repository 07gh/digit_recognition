Support Vector Machine (SVM) with one vs. all multiclassification
-- Gaussian kernel

One vs. all achieves 91.93% accuracy.
15% of training dataset (train.csv) used as validation set.
30% of training dataset (train.csv) used as training set.
This means that 55% of training dataset was unused. However, testing set remained constant through testing on every algorithm. Therefore, this accuracy can be compared to the accuracies of other algorithms.
classify_digits_gaussian_one_vs_all.m runs the classification.

Learn the model on SVM_gaussian_one_vs_all_learn.m. Automatically adds a column of ones
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
	kernel_weight_vector -- The vector of kernel weights of each predictor for each class
		kernel_weight_vector(i, j) = coefficient of datapoint i for class j
	X_kernel_vector -- Vector (list) of X_kernel points to be used for classification.
		X_kernel_vector(i, j, k) = Datapoint i, predictor j, for class k
	gamma -- Coefficient to be used in kernel calculation.

Once you've learned the model, test new observations using SVM_gaussian_one_vs_all_classify.m
Inputs:
	X -- Data to be classified. Same format as X in SVM_gaussian_one_vs_all_learn.m (Do not include column of ones)
	kernel_weight_vector -- Output from SVM_gaussian_one_vs_all_learn.m
	gamma -- Output from SVM_gaussian_one_vs_all_learn.m
	classes -- Unique classes.

Outputs:
	classifications -- classes predicted for X
		classifications(i) is the class predicted for observation i in X

SVM_gaussian_one_vs_all_cross_val.m. Helper function for SVM_gaussian_one_vs_all_learn.m. Runs cross validation on C.
Inputs:
	X_train -- Same as in SVM_gaussian_one_vs_all_learn.m
	y_train -- Same as in SVM_gaussian_one_vs_all_learn.m
	X_val -- Same as in SVM_gaussian_one_vs_all_learn.m
	y_val -- Same as in SVM_gaussian_one_vs_all_learn.m

Outputs:
	kernel_weights -- Kernel weights for X_kernel.
	X_kernel -- Set used for training. Used in calculation of kernel for classification.
	gamma -- Same as in SVM_gaussian_one_vs_all_classify.m
	classes -- For recall in classification