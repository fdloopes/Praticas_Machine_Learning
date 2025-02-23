%
% Implementation carried out during the Machine Learning course, 
% offered by Stanford University and available on the Coursera platform.
%
% Date: 12/03/2021
%
% This program aims to implement a logistic regression in a microchip 
% quality test data set, in order to be able to predict whether a 
% microchip will be approved or not.
%
% X(1) refers to test 1
% X(2) refers to test 2
% y refers to approval or not
%

%% Initialization
clear ; close all; clc
addpath('../helpers');

%% Load Data
%  The first two columns contains the X values and the third column
%  contains the label (y).

data = load('dataset.txt');
X = data(:, [1, 2]); 
y = data(:, 3);

plotData(X, y);

% Put some labels
hold on;

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

% Specified in plot order
legend('Accepted', 'Rejected')
hold off;


%% =========== Part 1: Regularized Logistic Regression ============

% Add Polynomial Features

% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled

X = mapFeature(X(:,1), X(:,2));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunction(initial_theta, X, y, lambda);

fprintf('\nCost at initial theta (zeros): %f\n', cost);
fprintf('Expected cost (approx): 0.693\n');
fprintf('\nGradient at initial theta (zeros) - first five values only:\n');
fprintf(' %f \n', grad(1:5));
fprintf('\nExpected gradients (approx) - first five values only:\n');
fprintf(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% Compute and display cost and gradient
% with all-ones theta and lambda = 10
test_theta = ones(size(X,2),1);

[cost, grad] = costFunction(test_theta, X, y, 10);

fprintf('\nCost at test theta (with lambda = 10): %f\n', cost);
fprintf('Expected cost (approx): 3.16\n');
fprintf('\nGradient at test theta - first five values only:\n');
fprintf(' %f \n', grad(1:5));
fprintf('\nExpected gradients (approx) - first five values only:\n');
fprintf(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ============= Part 2: Regularization and Accuracies =============
%
%  How does the decision boundary change when you vary lambda? How does
%  the training set accuracy vary?
%

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 1;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 50);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunction(t, X, y, lambda)), initial_theta, options);

% Plot Boundary
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('Accepted', 'Rejected', 'Decision boundary')
hold off;

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('\nTrain Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('Expected accuracy (with lambda = 1): 83.1 (approx)\n\n');

