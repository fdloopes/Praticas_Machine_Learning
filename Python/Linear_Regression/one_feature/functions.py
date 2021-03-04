#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 23:51:03 2021

@author: fdlopes

The file contains the implementations of the linear regression functions: 
Cost and Gradient Descent. Both functions were implemented to work with a 
single variable.

"""
# imports
import numpy as np

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
    m = y.size
    J_history = np.zeros(num_iters)

    for iter in range(num_iters):

        h = np.dot(theta,X)    # Calculate h_theta(x)
        error = (h - y)        # Calculate Error
    
        theta1 = theta[0] - alpha * (1/m * sum(error * X[0]))  # Calculate theta1
        theta2 = theta[1] - alpha * (1/m * sum(error * X[1]))  # Calculate theta2
    
        theta = [theta1,theta2]  # Update theta
    
    # ============================================================

    # Save the cost J in every iteration    
    J_history[iter] = costFunction(X, y, theta)
    
    return theta
