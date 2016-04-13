% Runs fake data on LDA.m

X = [1,1;1,2;3,2;2,3;-1,-3;-2,-3;2,4;-3,-2;.5,.5];
y = [1;1;1;1;-1;-1;-1;-1;-1];
[means, C_inv, prob, classes] = LDA(X, y);
classify_LDA([2, 3; -2, -1], means, C_inv, prob, classes)