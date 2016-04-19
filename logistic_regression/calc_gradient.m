function grad = calc_gradient(beta, X, y)
    [n, p] = size(X);
    %value = 0;
    grad = zeros(p, 1);
    
    for i=1:n
       %value = value + log(1+exp(y(i)*X(i,:)*beta));
       grad = grad-y(i)*X(i,:)'/(1+exp(y(i)*X(i,:)*beta));
    end
end