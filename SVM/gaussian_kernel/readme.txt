Support Vector Machine (SVM) with Gaussian (radial) kernel

classify_digits_gaussian.m runs the classification on 2 classes. For > 2 classes, use the one_vs_all classification. Runs cross-validation on C.

NB: Do not include a column of ones in X. The functions add it in automatically.

Learn the model on SVM_gaussian.m
Inputs:
	X -- Input matrix of size n x p
		n -- Number of observations.
		p -- Number of predictors.
		X(i, j) =  predictor value j of observation i
	y -- Class corresponding to X. Size n x 1
		The value in row i of y is the class of observation i in X
	C -- Error penalty

Outputs:
	kernel_weights -- Coefficients for SVM, including intercept in first spot.
	X_ones -- Used in classification with the kernel function.
	classes -- The 2 classes.
		SVM uses classes 1 and -1 for its algorithms, but user can use any two classes without any problems because the functions automatically convert to classes 1 and -1.
	gamma -- Gamma value used. Variance of entire dataset is used for sigma^2. (gamma = -.5/	sigma^2)

Once you've learned the model, test new observations using classify_SVM_gaussian.m
Inputs:
	X -- Data to be classified. Same format as X in SVM_gaussian.m (Do not add column of ones)
	kernel_weights -- Output from SVM_gaussian.m
	classes -- Output from SVM_gaussian.m
	X_kernel -- X from the learning (training) set. Output from SVM_gaussian.m
	gamma -- Output from SVM_gaussian.m

Outputs:
	classifications -- classes predicted for X
		classifications(i) is the class predicted for observation i in X

calc_distance_from_hyperplane_gaussian.m is a helper function for classify_SVM_gaussian.m
It calculates the distance from the hyperplane for the given observation
Inputs:
	x -- Observation to calculate distance from hyperplane. One row in X from classify_SVM_gaussian.m
	kernel_weights -- Same as classify_SVM_gaussian.m
	X_kernel -- Same as classify_SVM_gaussian.m
	gamma -- Same as classify_SVM_gaussian.m

Outputs:
	dist -- Distance from hyperplane, including direction (positive is 'above' hyperplane, while negative is 'below' hyperplane)

kernel.m is a helper function used to implement the kernel function (in this case Gaussian kernel)
Inputs:
	x -- First data point
	z -- Second data point
	gamma -- Coefficient. Same as classify_SVM_gaussian

Outputs:
	val -- Value of the kernel function for given two points.