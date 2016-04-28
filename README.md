# digit_recognition
Hand-written digit recognition for Machine Learning class

Data obtained at http://statweb.stanford.edu/~tibs/ElemStatLearn/data.html

Currently have classification methods for:  
--Linear Discriminant Analysis (LDA), including multi-class (accuracy 87.24%)  

--Logistic Regression (logit), including:  
----simple logistic regression (no regularization) (accuracy 85.55%)  
----ridge penalty regularization for multi-class (accuracy 89.19%)  
----lasso penalty regularization for multi-class (accuracy 89.74%) 

--Support Vector Machine (SVM), including:  
----linear kernel (accuracy 89.12%)
----Gaussian (radial) kernel (accuracy 90.98%)