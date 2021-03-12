function [J, grad] = costFunction(theta, X, y, lambda)
%
% Computes the cost of using theta as the parameter for regularized 
% logistic regression and the gradient of the cost.
% 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

h_theta = sigmoid(X*theta);   % Calculate h_theta(X);

reg = (lambda/(2*m)) * sum(theta(2:end).^2); % Calculate regularized term

J = (1/m)*sum(-y.*log(h_theta) - (1-y).*log(1-h_theta)) + reg;  % Calculate de cost

grad(1) = (1/m)*(X(:,1)'*(h_theta - y));  % Calculate gradient descent position zero

grad(2:end) = (1/m)*(X(:,2:end)'*(h_theta - y)) + ((lambda/m)*theta(2:end)); % Calculate gradient descent

% =============================================================

end
