Linear Discriminant Analysis

Can support multiple (>2) classes

Learn the model on LDA.m
Inputs:
	X -- Input matrix of size n x p
		n -- Number of observations.
		p -- Number of predictors.
		X(i, j) =  predictor value j of observation i
	y -- Class corresponding to X. Size n x 1
		The value in row i of y is the class of observation i in X

Outputs:
	means -- The mean value of each predictor for each class
		means(i, j) = mean of predictor j for class i
	C_inv -- Inverse of correlation matrix for X
	prob -- Frequency of each class in X. Assumed to be the prior probability of each class
		prob(i) = frequency of class i in X
	classes -- All classes represented by y
		The index of each class corresponds to the index in other outputs.

Once you've learned the model, test new observations using classify_LDA.m
Inputs:
	X -- Data to be classified. Same format as X in LDA.m
	means -- Output from LDA.m
	C_inv -- Output from LDA.m
	prob -- Output from LDA.m
	classes -- Output from LDA.m

Outputs:
	classifications -- classes predicted for X
		classifications(i) is the class predicted for observation i in X

calc_LDA_discriminant.m is a helper function for classify_LDA.m
It calculates the discriminant for the given class for the given observationInputs:
	x -- Observation to calculate disciminant function for. One row in X from classify_LDA.m
	means -- Same as LDA.m
	C_inv -- Same as LDA.m
	prob -- Same as LDA.m
	class -- Class to calculate discriminant function for. One row in classes from classify_LDA

Outputs:
	disc -- Discriminant value for class for observation x
