function p = predict(theta, X)
%
% Computes the predictions for X using a threshold at 0.5 
%(i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
%

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

p = round(sigmoid(X*theta)); % Calculate predict, 0 or 1

% =========================================================================


end
