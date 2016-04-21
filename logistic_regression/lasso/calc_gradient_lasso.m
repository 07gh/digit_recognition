function grad = calc_gradient_lasso(beta, X, y)
    [n, p] = size(X);
    grad = zeros(p, 1);
    
    for i=1:n
       grad = grad-y(i)*X(i,:)'/(1+exp(y(i)*X(i,:)*beta));
    end
end