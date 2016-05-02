# digit_recognition
Hand-written digit recognition for Machine Learning class

Best algorithm is SVM with Gaussian kernel (accuracy 91.93%)

Data obtained at http://statweb.stanford.edu/~tibs/ElemStatLearn/data.html

Currently have classification methods for:  
-- Linear Discriminant Analysis (LDA), including multi-class (accuracy 87.24%)  
---- Runs relatively quickly  

-- Logistic Regression (logit), including:  
---- Simple logistic regression (no regularization) (accuracy 85.55%)  
------ Runs relatively quickly  
---- Ridge penalty regularization for multi-class (accuracy 89.19%)  
------ Takes a few hours to run  
---- Lasso penalty regularization for multi-class (accuracy 89.74%)  
------ Takes a few hours to run  

-- Support Vector Machine (SVM), including:  
---- Linear kernel (accuracy 89.12%)  
------ Takes a few hours to run  
---- Gaussian (radial) kernel (accuracy 91.93%)  
------ Takes multiples hours to run