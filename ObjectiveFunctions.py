#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 03 11:26:52 2019

@author: 33fred33
"""

def mse(y, y_predicted):
    """
    Positional arguments:
        y is a list of labels
        y_predicted is a predicted list of labels
    Returns: Mean Squared Error as a float
    """
    n = len(y)
    MSE = sum([pow(y[i]-y_predicted[i],2) for i in range(n)]) / n
    return MSE

def single_goal_accuracy(y, y_predicted, goal_class):
    """
    Positional arguments:
        y is a list of labels
        y_predicted is a predicted list of labels
        goal_class is the label to be compared
    Returns: y_predicted match ration with y, over the given goal class only, as a single float 
    """
    corrects = sum([1 if y_predicted[i] == y[i] and y[i] == goal_class else 0 for i in range(len(y))])
    total = sum([1 if y[i] == goal_class else 0 for i in range(len(y))])
    if total == 0:
    	print("goal_class", goal_class)
    	print("y_predicted", y_predicted)
    	print("y", y)
    accuracy = corrects / total
    
    return accuracy

def accuracy(y, y_predicted):
    """
    Positional arguments:
        y is a list of labels
        y_predicted is a predicted list of labels
    Returns: accuracy
    """
    corrects = sum([1 if y_predicted[i] == y[i] else 0 for i in range(len(y))])
    accuracy = corrects / len(y)
    return accuracy

