function dist = calc_distance_from_hyperplane_gaussian(x, kernel_weights, X_kernel, gamma)
    dist = 0;
    [n, ~] = size(X_kernel);
    for i=1:n
        dist = dist + kernel_weights(i) * kernel(x, X_kernel(i, :), gamma);
    end
end