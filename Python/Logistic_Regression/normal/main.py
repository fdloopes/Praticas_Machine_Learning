#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 22:18:53 2021

@author: fdlopes

This program aims to implement a logistic regression in a data set of student 
grades, in order to be able to predict whether a student will pass or not.

X(1) refers to grade 1
X(2) refers to grade 2
y refers to approval or not

"""

# imports

import sys
sys.path.append('../')
import numpy as np
import csv
import matplotlib.pyplot as plt
import scipy.optimize as op
from functions import costFunction, gradientDescent, sigmoid, predict
import helpers.functions as hlp

# Load dataset
with open('dataset.csv',newline='') as f:
    reader = csv.reader(f,delimiter=',')
    data = list(reader)


# Initialization
X = np.array([np.array(data).transpose()[0],np.array(data).transpose()[1]])# Decompose the data array
y = np.array(data).transpose()[2]   # Decompose the data array, approval or not 
m = y.size                          # Number of training examples

# Convert data to float
X = X.astype(np.float)
y = y.astype(np.float)


# ==================== Part 1: Plotting ====================

plt.rcParams['figure.figsize'] = (11,7)

labels = ['Admitted','Not admitted','Exam 1 score','Exam 2 score']
hlp.plotData(X,y,labels)
plt.legend(loc='upper right')
plt.show()

# ============ Part 2: Compute Cost and Gradient ============

# Setup the data matrix appropriately, and add ones for the intercept term
n,m = X.shape

# Add intercept term to x and X_test
X = np.array([np.ones(m),X[0],X[1]])

# Initialize fitting parameters
initial_theta = np.zeros(n + 1)

# Compute and display initial cost and gradient
cost = costFunction(initial_theta, X, y)
grad = gradientDescent(initial_theta, X, y)

print('\nCost at initial theta (zeros): ', cost)
print('\nExpected cost (approx): 0.693')
print('\nGradient at initial theta (zeros): \n', grad)
print('\nExpected gradients (approx):\n [-0.1000 -12.0092 -11.2628]')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2])
cost = costFunction(test_theta, X, y)
grad = gradientDescent(test_theta, X, y)

print('\nCost at test theta: ', cost)
print('\nExpected cost (approx): 0.218')
print('\nGradient at test theta: \n', grad)
print('\nExpected gradients (approx):\n [0.043 2.566 2.647]\n')

# ============= Part 3: Optimizing using optimize.minimize  =============

#  Set options for minimize
options = {'maxfun':400}

ret = op.minimize(fun=costFunction,x0=initial_theta, args=(X,y),
                  method='TNC',jac=gradientDescent,options=options)

cost = ret.fun
theta = ret.x

# Print theta to screen
print('Cost at theta found by minimize function:\n', cost)
print('\nExpected cost (approx): 0.203')
print('\ntheta: ', theta)
print('\nExpected theta (approx): [-25.161 0.206 0.201]')

# Plot the boundary line
hlp.plotData([X[1],X[2]],y,labels)
hlp.plotBoundary(X,theta)
plt.legend(loc='lower left')
plt.show()

# ============== Part 4: Predict and Accuracies ==============

#  Predict probability for a student with score 45 on exam 1 
    #  and score 85 on exam 2 

prob = sigmoid(np.dot([1,45,85], theta))

# Plot data + student score

hlp.plotData([X[1],X[2]],y,labels)
hlp.plotBoundary(X, theta)
plt.scatter(45,85,label="Predict",c='g',marker='s')
plt.legend(loc='lower left')
plt.show()

print('\nFor a student with scores 45 and 85, we predict an admission probability of ', prob)
         
print('\nExpected value: 0.775 +/- 0.002\n')

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: ', np.mean(np.double(p == y)) * 100)
print('\nExpected accuracy (approx): 89.0\n')