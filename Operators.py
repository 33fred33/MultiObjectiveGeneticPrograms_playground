#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 03 11:26:52 2019

@author: 33fred33

"""

def safe_divide_numerator(a, b):
    """
    Positional arguments:
        a is a number
        b is a number
    Executes a/b. If b=0, returns a
    """
    if b == 0 :
        return a
    else:
        return a/b

def safe_divide_zero(a, b):
    """
    Positional arguments:
        a is a number
        b is a number
    Executes a/b. If b=0, returns 0
    """
    if b == 0 :
        return 0
    else:
        return a/b

def signed_if(condition, a, b):
    """
    Positional arguments:
        condition is a number
        a is a number
        b is a number
    Returns a if condition <= 0, b otherwise
    """
    if condition <= 0 :
        return a
    else:
        return b

def b_and(a, b):
    return a and b

def b_or(a, b):
    return a or b

def b_not(a):
    return not a

def b_if(condition, a, b):
    """
    Condition is a boolean
    """
    if condition:
        return a
    else:
        return b