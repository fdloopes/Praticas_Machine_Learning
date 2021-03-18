#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 13:15:52 2021

@author: fdlopes

The file contains the implementations of logistic regression functions:
cost function and gradient descent

"""
# imports

import numpy as np
from helpers.functions import sigmoid, predict


def costFunction(theta,X,y,lambdaa):
    
    # Initialize some useful values
    m = y.size
    
    # You need to return the following variables correctly 
    J = 0
    
    # Calculate h_theta(X)
    h_theta = sigmoid(np.dot(X,theta))
    
    # Calculate regularized term
    reg = (lambdaa/(2*m)) * sum(theta[1:m+1]**2)
    
    # Calculate de cost
    J = ((1/m)*sum(-y*np.log(h_theta) - (1-y)*np.log(1-h_theta))) + reg
    
    return J

def gradientDescent(theta,X,y,lambdaa):
    # Initialize some useful values
    m = y.size    
    grad = np.zeros(theta.size)
    
    # Calculate h_theta(X)
    h_theta = sigmoid(np.dot(X,theta))
    
    # Calculate gradient descent position zero
    grad[0] = (1/m)*np.dot(X[:,0],(h_theta - y))  
    
    # Calculate gradient descent
    grad[1:m+1] = (1/m)*np.dot((h_theta - y),X[:,1:m+1]) + ((lambdaa/m) * theta[1:m+1]) 
    
    return np.around(grad,4)
