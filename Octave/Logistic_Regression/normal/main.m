%
% Implementation carried out during the Machine Learning course, 
% offered by Stanford University and available on the Coursera platform.
%
% Date: 12/03/2021
%
% This program aims to implement logistic regression on a student exam data set, 
% in order to be able to predict whether a student will pass or not.
%
% X(1) refers to exam 1
% X(2) refers to exam 2
% y refers to approval or not
%

%% Initialization
clear ; close all; clc
addpath("../helpers");

%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.

data = load('dataset.txt');
X = data(:, [1, 2]); 
y = data(:, 3);

%% ==================== Part 1: Plotting ====================

fprintf(['\nPlotting data with + indicating (y = 1) examples and o ' ...
         'indicating (y = 0) examples.\n']);

plotData(X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;

%% ============ Part 2: Compute Cost and Gradient ============

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

fprintf('\nCost at initial theta (zeros): %f\n', cost);
fprintf('Expected cost (approx): 0.693\n');
fprintf('\nGradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);
fprintf('\nExpected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n');

% Compute and display cost and gradient with non-zero theta
test_theta = [-24; 0.2; 0.2];
[cost, grad] = costFunction(test_theta, X, y);

fprintf('\nCost at test theta: %f\n', cost);
fprintf('Expected cost (approx): 0.218\n');
fprintf('\nGradient at test theta: \n');
fprintf(' %f \n', grad);
fprintf('\nExpected gradients (approx):\n 0.043\n 2.566\n 2.647\n');

%% ============= Part 3: Optimizing using fminunc  =============
%  In this part, will use a built-in function (fminunc) to find the
%  optimal parameters theta.

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Print theta to screen
fprintf('\nCost at theta found by fminunc: %f\n', cost);
fprintf('Expected cost (approx): 0.203\n');
fprintf('\ntheta: \n');
fprintf(' %f \n', theta);
fprintf('\nExpected theta (approx):\n');
fprintf(' -25.161\n 0.206\n 0.201\n');

% Plot Boundary
plotDecisionBoundary(theta, X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted', 'Decision boundary')
hold off;

%% ============== Part 4: Predict and Accuracies ==============

%  Predict probability for a student with score 45 on exam 1 
%  and score 85 on exam 2 

prob = sigmoid([1 45 85] * theta);

fprintf(['\nFor a student with scores 45 and 85, we predict an admission ' ...
         'probability of %f\n'], prob);
         
fprintf('Expected value: 0.775 +/- 0.002\n\n');

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('Expected accuracy (approx): 89.0\n');
fprintf('\n');
