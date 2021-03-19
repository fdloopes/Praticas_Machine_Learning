#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 20:14:38 2021

@author: fdlopes

This program aims to implement linear regression on a set of food truck efficiency 
data by city,to be able to predict where it will be more profitable to open a new 
food truck.

X refers to the population size in 10,000s
y refers to the profit in $10,000s

"""

# imports
import numpy as np
import csv
import matplotlib.pyplot as plt
from functions import costFunction, gradientDescent

# Load dataset
with open('dataset.csv',newline='') as f:
    reader = csv.reader(f,delimiter=',')
    data = list(reader)


# Initialization
X = np.array(data).transpose()[0]   # Decompose the data array, get population
y = np.array(data).transpose()[1]   # Decompose the data array, get profit
m = y.size                           # Number of training examples

# Convert data to float
X = X.astype(np.float)
y = y.astype(np.float)

# Plot data
plt.rcParams['figure.figsize'] = (11,7)
plt.scatter(X,y,label='Training data',marker='x',c='g')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')

## =================== Part 1: Cost and Gradient descent ===================

X = [np.ones(m),X] # Add a column of ones to x

theta = np.zeros(2) # Initializing fit parameters

iterations = 1500
alpha = 0.01

print('\nTesting the cost function ...\n')

# Compute and display initial cost
J = costFunction(X, y, theta)
    
print('With theta = [0 ; 0]\nCost computed =', J)
print('Expected cost value (approx) 32.07\n')

# Further testing of the cost function
J = costFunction(X, y, [-1 , 2])

print('With theta = [-1 ; 2]\nCost computed =', J)
print('Expected cost value (approx) 54.24\n')

# Run gradient descent
print('Running Gradient Descent ...\n')
theta = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent:')
print(round(theta[0],4),'  ',round(theta[1],4))
print('\nExpected theta values (approx):')
print('-3.6303    1.1664\n')

# Plot the linear fit
plt.plot(X[1], np.dot(theta,X), 'r-',label='Linear regression')
plt.legend(loc='upper left')
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = round(np.dot([1, 35.0],theta),4)
print('For population = 35,000, we predict a profit of', predict1)

predict2 = round(np.dot([1, 70.0], theta),4)
print('For population = 70,000, we predict a profit of', predict2)
