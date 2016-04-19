function grad = calc_gradient_ridge(beta, lambda, X, y)
    [n, ~] = size(X);
    grad = lambda * beta;
    
    for i=1:n
       grad = grad-y(i)*X(i,:)'/(1+exp(y(i)*X(i,:)*beta));
    end
end