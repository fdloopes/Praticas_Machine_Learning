#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 00:53:32 2021

@author: fdlopes

The file contains the implementations of the linear regression functions: 
Cost, Gradient Descent and Normal Equation. All functions were implemented to
work with a multi variables.

"""

import numpy as np

def featureNormalize(X):
    
    # Initialize some useful values
    X_norm = X
    mu = np.zeros(np.shape(X)[0])
    sigma = np.zeros(np.shape(X)[0])
    
    # Formula X[i] = X[i] - mean(X[i])/std_deviation(X[i])
    mu = round(np.mean(X[0,:]),4),round(np.mean(X[1,:]),4)    # Mean value of each feature
    sigma = [round(np.std(X[0,:]),4),round(np.std(X[1,:]),4)]  # standard deviation of each feature
    
    for i in range(np.shape(X)[1]):
        X_norm[:,i] = (X[:,i] - mu)/ sigma # Features of X matriz normalization
    
    return [X, mu, sigma]

def costFunction(X,y,theta):
    
    # Initialize some useful values
    m = y.size    
    J = 0
    
    # Formula 1/2*m * sum[i=1:m](h_theta(x[i]) - y[i])²
    h = np.dot(theta,X)         # Calculate h_theta(x)
    error = sum((h - y)**2)     # Calculate Square Error
    J = error * (1/(2*m))       # Multiply by the constant and return the cost
    
    return J


def gradientDescent(X,y,theta,alpha,num_iters):
    
    # Initialize some useful values
    m = y.size # number of training examples
    J_history = np.zeros(num_iters)
    
    # Formula theta[j] = theta[j] - alpha * 1/m * sum[i=1:m]((h_theta(x[i])-y[i])*x[i][j]
    for iter in range(num_iters):
        
        # Formula h_theta(x) = theta0 * X0 + theta1 * X1 + ... + thetaN * Xn
        h = np.dot(theta,X)    # Calculate h_theta(x)
        error = (h - y)       # Calculate Error
        
        for j in range(int(np.shape(theta)[0])):
            theta[j] = theta[j] - (alpha * ((1/m) * sum(X[j] * error)))  # Calculate theta
       
        # ============================================================

        # Save the cost J in every iteration    
        J_history[iter] = costFunction(X, y, theta)
    
    return [theta,J_history]

def normalEqn(X,y):
    
    # Computes the closed-form solution to linear regression using the normal equations.
    theta = np.zeros(2)
    
    # Formula theta = inverse(X' * X) * (X' * y)
    theta = np.dot(y, (np.dot(X.T, np.linalg.inv(np.dot(X,X.T)))))  
    
    return theta