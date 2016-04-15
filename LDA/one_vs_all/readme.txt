Linear Discriminant Analysis with one vs. all multiclassification

Ran to test one vs. all techniques

One vs. all achieces 87.10% accuracy, less than basic LDA
classify_digits_one_vs_all.m runs the classification.

Learn the model on LDA_one_vs_all_learn.m
Inputs:
	X -- Input matrix of size n x p
		n -- Number of observations.
		p -- Number of predictors.
		X(i, j) =  predictor value j of observation i
	y -- Class corresponding to X. Size n x 1
		The value in row i of y is the class of observation i in X

Outputs:
	mean_vector -- The vector of mean values of each predictor for each class
		mean_vector(i, j, k) = mean of predictor j for class k vs others (i==1 is class k, i==0 is class not k)
	C_inv_vector -- Vector for inverse of correlation matrix for X
	prob_vector -- Vector of frequencies of each class in X. Assumed to be the prior probability of each class
		prob(i, k) = frequency of class k in X (i==1 is class k, i==0 is class not k)
	classes -- All classes represented by y
		The index of each class corresponds to the index in other outputs.

Once you've learned the model, test new observations using LDA_one_vs_all_classify.m
Inputs:
	X -- Data to be classified. Same format as X in LDA_one_vs_all_learn.m
	mean_vector -- Output from LDA_one_vs_all_learn.m
	C_inv_vector -- Output from LDA_one_vs_all_learn.m
	prob_vector -- Output from LDA_one_vs_all_learn.m
	classes -- Output from LDA_one_vs_all_learn.m

Outputs:
	classifications -- classes predicted for X
		classifications(i) is the class predicted for observation i in X