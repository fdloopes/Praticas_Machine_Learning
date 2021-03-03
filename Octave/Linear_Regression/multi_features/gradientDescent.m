function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

% Updates theta by taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    h = X * theta;    % Calculate h_theta(x)
    error = (h - y);  % Calculate Error
    
    for j = 1:size(theta)
      theta(j) = theta(j) - alpha * (1/m * sum(error .* X(:,j)));  % Calculate theta
    endfor
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = costFunction(X, y, theta);

end

end
