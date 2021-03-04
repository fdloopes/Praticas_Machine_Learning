function J = costFunction(X, y, theta)
% Computes the cost of using theta as the parameter for linear regression 
% to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

h = X * theta;              % Calculate h_theta(X)
error = sum((h - y) .^ 2);  % Calculate Square Error
J = error * (1/(2*m));      % Multiply by the constant and return the cost.

% =========================================================================

end
