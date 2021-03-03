%
% Implementation carried out during the Machine Learning course, 
% offered by Stanford University and available on the Coursera platform.
%
% Date: 03/03/2021
%
% This program aims to implement a linear regression in a set of property 
% price data by city, in order to be able to predict how much the value of 
% each property will be according to the size and number of rooms.
%
% X(1) refers to the size of house in square feet
% X(2) refers to the number of bedrooms
% y refers to the profit, price of houses
%

%% Initialization
clear ; close all; clc

addpath("../helpers");

%% ================ Part 1: Feature Normalization ================

fprintf('Loading data ...\n');

%% Load Data
data = load('dataset.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('\nNormalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

%% ================ Part 2: Gradient Descent ================

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.1;
num_iters = 550;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);

% Run gradient descent
[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
hold on;

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 bedrooms house
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.

price = 0; % You should change this

house = [1, 1650, 3];

house(2:3) = (house(2:3) - mu(1:2)) ./ sigma(1:2); % Features normalization

price = house * theta; # Prediction price

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================

fprintf('\nSolving with normal equations...\n');

%% Load Data
data = load('dataset.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house

price = 0; % You should change this

house = [1, 1650, 3];

price = (house) * theta; # Prediction price

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);

fprintf(['\nComparing the gradient descent with the normal equation it is possible to notice that both obtained the same result,' ...
        'however the normal equation was much more efficient and needed less code,' ... 
        'since the gradient descent required 550 iterations to reach the same result of the normal equation.\n\n']);
