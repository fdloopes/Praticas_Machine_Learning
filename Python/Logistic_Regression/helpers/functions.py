#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:35:12 2021

@author: fdlopes

The file contains the implementations of auxiliary logistic regression functions:
predict, sigmoid, mapFeature, plotData and plotBoundary

"""

import numpy as np
import matplotlib.pyplot as plt

def predict(theta, X):
    
    # Initialize some useful values
    m = X.size
    
    p = np.zeros(m)
    
    # Calculate predict, 0 or 1
    p = np.round(sigmoid(np.dot(theta,X))) 
    
    return p

def sigmoid(z):
    
    # Initialize some useful values
    g = np.zeros(z.size);
    
    # Calculate sigmoid
    g = 1 / (1+np.exp(-z)) 

    return g

# Maps the two input features to quadratic features used in the
#
# Returns a new feature array with more features, comprising of 
# X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
#
# Inputs X1, X2 must be the same size
#
def mapFeature(X1,X2):
    
    degree = 6
    collumn = 1
    out = np.ones((X1.size, sum(range(degree+2))))
    for i in range(1,degree+1):
        for j in range(0,i+1):
            out[:,collumn] = (X1**(i-j))*(X2 ** j)
            collumn += 1
    return out


def plotData(X,y,labels):
    
    # Find Indices of Positive and Negative Examples
    pos = np.array(np.where(y==1)).transpose()
    neg = np.array(np.where(y==0)).transpose()
    
    # Plot data    
    p1 = plt.scatter(X[0][pos],X[1][pos],label=labels[0],marker='x',c='b')
    p2 = plt.scatter(X[0][neg],X[1][neg],label=labels[1],marker='o',c='r')
    plt.xlabel(labels[2])
    plt.ylabel(labels[3])
    return p1,p2
    
def plotBoundary(X,theta):
    
    if X.shape[1] <= 2 :
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([min(X[1,:])-2,  max(X[1,:])+2])
        
        # Calculate the decision boundary line
        plot_y = (-1/theta[2])*(theta[1]*plot_x + theta[0])
        
        # Plot boundary line
        plt.plot(plot_x,plot_y,label='Decision Boundry',c='k')
        
    else:
        # Here is the grid range
        u = np.linspace(-0.75, 1.2, 50)
        v = np.linspace(-0.75, 1.2, 50)
    
        z = np.zeros(( len(u), len(v) ))
        
        # Evaluate z = theta*x over the grid
        for i in range(1,u.shape[0]):
            for j in range(1,v.shape[0]):
                z[i,j] = np.dot(mapFeature(u[i], v[j]),theta)
        
        
        z = z.T # important to transpose z before calling contour
        
        return plt.contour(u, v, z, levels=[0])
        