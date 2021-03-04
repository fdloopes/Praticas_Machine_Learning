#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 00:50:26 2021

@author: fdlopes

This program aims to implement a linear regression in a set of property 
price data by city, in order to be able to predict how much the value of 
each property will be according to the size and number of rooms.

X(1) refers to the size of house in square feet
X(2) refers to the number of bedrooms
y refers to the profit, price of houses

"""

# imports
import numpy as np
import csv
import matplotlib.pyplot as plt
from functions import featureNormalize, costFunction, gradientDescent, normalEqn

# Load dataset
with open('dataset.csv',newline='') as f:
    reader = csv.reader(f,delimiter=',')
    data = list(reader)


# Initialization
X = np.array([np.array(data).transpose()[0],np.array(data).transpose()[1]])# Decompose the data array
y = np.array(data).transpose()[2]   # Decompose the data array, get prices 
m = y.size                          # Number of training examples

# Convert data to float
X = X.astype(np.float)
y = y.astype(np.float)

# Scale features and set them to zero mean
print('\nNormalizing Features ...\n')

[X, mu ,sigma] = featureNormalize(X)

X = [np.ones(m),X[0],X[1]]
## ================ Part 1: Gradient Descent ================

print('Running gradient descent ...\n')

# Choose some alpha value
alpha = 0.1
num_iters = 550

# Init Theta and Run Gradient Descent 
theta = np.zeros(3)

# Run gradient descent
[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.plot(range(J_history.size), J_history, '-b', 'LineWidth', 2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

# Display gradient descent's result
print('Theta computed from gradient descent: \n')
print(theta)
print('\n')

# Estimate the price of a 1650 sq-ft, 3 bedrooms house
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.

price = 0 # You should change this

house = [1, 1650, 3]

house[1] = (house[1] - mu[0]) / sigma[0] # Features normalization
house[2] = (house[2] - mu[1]) / sigma[1] # Features normalization

price = np.dot(house,theta) # Prediction price

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):', price)
# ============================================================

# ================ Part 2: Normal Equations ================

print('\nSolving with normal equations...\n')

# Load dataset
with open('dataset.csv',newline='') as f:
    reader = csv.reader(f,delimiter=',')
    data = list(reader)


# Initialization
X = np.array([np.array(data).transpose()[0],np.array(data).transpose()[1]])# Decompose the data array
y = np.array(data).transpose()[2]   # Decompose the data array, get prices 
m = y.size                          # Number of training examples

# Convert data to float
X = X.astype(np.float)
y = y.astype(np.float)


# Add intercept term to X
X = np.stack([np.ones(m),X[0],X[1]])

# Calculate the parameters from the normal equation
theta = normalEqn(X, y)

# Display normal equation's result
print('Theta computed from the normal equations: \n')
print(theta)
print('\n')

# Estimate the price of a 1650 sq-ft, 3 br house

price = 0 # You should change this

house = [1, 1650, 3]

price = np.dot(house,theta) # Prediction price

print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):', price)


# ============================================================
