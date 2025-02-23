function [X_norm, mu, sigma] = featureNormalize(X)
% Returns a normalized version of X where
% the mean value of each feature is 0 and the standard deviation
% is 1. This is often a good preprocessing step to do when
% working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

mu = mean(X);    % Mean value of each feature
sigma = std(X);  % standard deviation of each feature

for i = 1:length(X)
  X_norm(i,:) = (X(i,:) - mu) ./ sigma; % Features of X matriz normalization
endfor

% ============================================================

end
