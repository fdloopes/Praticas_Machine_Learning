#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 12:50:57 2021

@author: fdlopes

This program aims to implement a logistic regression in a microchip 
quality test data set, in order to be able to predict whether a 
microchip will be approved or not.

X(1) refers to test 1
X(2) refers to test 2
y refers to approval or not

"""

# imports
import sys
sys.path.append('../')
import csv
import numpy as np
import matplotlib.pyplot as plt
import helpers.functions as hlp
import scipy.optimize as op
from functions import costFunction, gradientDescent

# Load dataset
with open('dataset.csv',newline='') as f:
    reader = csv.reader(f,delimiter=',')
    data = list(reader)


# Initialization
# Decompose the data array
X = np.array([np.array(data).transpose()[0],np.array(data).transpose()[1]])
y = np.array(data).transpose()[2]   # Decompose the data array, approval or not 
m = y.size                          # Number of training examples

# Convert data to float
X = X.astype(np.float)
y = y.astype(np.float)

# ==================== Part 1: Plotting ====================

plt.rcParams['figure.figsize'] = (11,7)

labels = ['Accepted','Rejected','Microchip test 1','Microchip test 2']
hlp.plotData(X,y,labels)
plt.legend(loc='upper right')
plt.show()

# =========== Part 2: Regularized Logistic Regression ============

# Add Polynomial Features
X = hlp.mapFeature(X[0,:], X[1,:])

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1
lambdaa = 1

# Compute and display initial cost and gradient for regularized logistic
# regression

cost = costFunction(initial_theta, X, y, lambdaa)
grad = gradientDescent(initial_theta, X, y, lambdaa)

print('\nCost at initial theta (zeros):', cost);
print('\nExpected cost (approx): 0.693');
print('\nGradient at initial theta (zeros) - first five values only:\n', grad[0:5])
print('\nExpected gradients (approx) - first five values only:\n');
print('[0.0085 0.0188 0.0001 0.0503 0.0115]');

# Compute and display cost and gradient
# with all-ones theta and lambda = 10

lambdaa = 10
test_theta = np.ones(X.shape[1])

cost = costFunction(test_theta, X, y, lambdaa)
grad = gradientDescent(test_theta, X, y, lambdaa)

print('\nCost at test theta (with lambda = 10):\n', cost)
print('\nExpected cost (approx): 3.16')
print('\nGradient at test theta - first five values only:\n',grad[0:5])
print('\nExpected gradients (approx) - first five values only:\n')
print('[0.3460 0.1614 0.1948 0.2269 0.0922]\n')

# ============= Part 3: Regularization and Accuracies =============

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1]);

# Set regularization parameter lambda to 1 (you should vary this)
lambdaa = 1;

#  Set options for minimize
options = {'maxfun':60}

ret = op.minimize(fun=costFunction,x0=initial_theta, args=(X,y,lambdaa),
                  method='TNC',jac=gradientDescent,options=options)

cost = ret.fun
theta = ret.x

# Plot the boundary line
p1,p2, = hlp.plotData([X[:,1],X[:,2]],y,labels)
p3,_ = hlp.plotBoundary(X,theta).legend_elements()

# Create a legend
plt.legend([p1,p2,p3[0]],[labels[0],labels[1],'Boundary Line'], loc='lower left')

# Plot Data
plt.show()

# Compute accuracy on our training set
p = hlp.predict(theta, X.T)

print('Train Accuracy:', np.mean(p == y) * 100)
print('\nExpected accuracy (with lambda = 1): 83.1 (approx)\n\n')
