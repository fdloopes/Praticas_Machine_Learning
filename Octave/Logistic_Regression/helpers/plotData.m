function plotData(X, y)
%
% Plots the data points with + for the positive examples
% and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% Find Indices of Positive and Negative Examples
pos = find(y==1); 
neg = find(y==0);

% Plot Examples
plot(X(pos, 1), X(pos, 2), 'bx','LineWidth', 2, 'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'r','MarkerSize', 7);

% =========================================================================

hold off;

end
