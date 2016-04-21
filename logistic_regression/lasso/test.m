% Runs fake data on logistic_regression_lasso.m

X = [1,1;1,2;3,2;2,3;-1,-3;-2,-3;2,4;-3,-2;.5,.5];
y = [1;1;1;1;0;0;0;0;0];
beta = logistic_regression_lasso(X, y, 2)

n = size(y);

x_min = min(X(:, 1));
x_max = max(X(:, 2));
x = x_min:.01:x_max;
plot(x, -beta(1)/beta(3) - beta(2)/beta(3)*x);
hold on;
for i=1:n(1)
    if y(i) > 0
        plot(X(i, 1), X(i, 2),'*')
    else
        plot(X(i, 1), X(i, 2),'o')
    end
end