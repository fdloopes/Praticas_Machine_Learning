function [J, grad] = costFunction(theta, X, y)
  
% Computes the cost of using theta as the parameter for 
% logistic regression and the gradient of the cost.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

h_theta = sigmoid(X*theta);   % Calculate h_theta(X)

J = (1/m).*sum(-y.*log(h_theta) - (1-y).*log(1-h_theta));   % Calculate the cost

grad = (1/m).*sum((h_theta-y).*X);  % Calculate gradient descent

% =============================================================

end
