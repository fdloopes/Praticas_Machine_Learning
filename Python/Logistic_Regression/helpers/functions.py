#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:35:12 2021

@author: fdlopes

The file contains the implementations of auxiliary logistic regression functions:
predict, sigmoid, plotData and plotBoundary

"""

import numpy as np
import matplotlib.pyplot as plt

def predict(theta, X):
    
    
    # Initialize some useful values
    m = X.shape 
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

def plotData(X,y,labels):
    
    # Find Indices of Positive and Negative Examples
    pos = np.array(np.where(y==1)).transpose()
    neg = np.array(np.where(y==0)).transpose()
    
    # Plot data    
    plt.scatter(X[0][pos],X[1][pos],label=labels[0],marker='x',c='b')
    plt.scatter(X[0][neg],X[1][neg],label=labels[1],marker='o',c='r')
    plt.xlabel(labels[2])
    plt.ylabel(labels[3])
    
def plotBoundary(X,theta):

    # Only need 2 points to define a line, so choose two endpoints
    plot_x = np.array([min(X[1,:])-2,  max(X[1,:])+2])
    
    # Calculate the decision boundary line
    plot_y = (-1/theta[2])*(theta[1]*plot_x + theta[0])
    
    # Plot boundary line
    plt.plot(plot_x,plot_y,label='Decision Boundry',c='k')
    