#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 00:57:12 2021

@author: fdlopes

The file contains the implementations of the Logistic regression functions: 
Cost and Gradient Descent.

"""

import numpy as np
from helpers.functions import sigmoid, predict

def costFunction(theta,X,y):
    
    # Initialize some useful values
    m = y.size
    J = 0
    
    # Calculate h_theta(X)
    h_theta = sigmoid(np.dot(theta,X))   
    
    # Calculate the cost
    J = (1/m)*sum(-y*np.log(h_theta) - (1-y)*np.log(1-h_theta))   
    
    return J

def gradientDescent(theta,X,y):
    
    # Initialize some useful values
    m = y.size    
    grad = np.zeros(theta.size)
    
    # Calculate h_theta(X)
    h_theta = sigmoid(np.dot(theta,X))
    
    # Calculate gradient descent
    grad = (1/m) * np.dot(X, (h_theta-y))   
    
    return grad


