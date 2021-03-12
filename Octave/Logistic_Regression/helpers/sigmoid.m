function g = sigmoid(z)

% Computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

g = 1 ./ (1+exp(-z)); % Calculate sigmoid

% =============================================================

end
