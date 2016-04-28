function val = kernel(x, z, gamma)
    % kernel function -- Gaussian kernel
    val = exp(gamma * norm(x - z)^2);
end