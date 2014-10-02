# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 09:27:15 2012

@author: khrapov
"""

import numpy as np

# Geometric Brownian Motion characteristic function
def CFGBM(u, P):
    # u is N-vector
    # T,r,sigma,nu,theta are scalars
    # returns N-vector
    T = P['T']
    r = P['r']
    sigma = P['sigma']
    
    phi = np.exp(u * r * T * 1j - u**2 * sigma**2 * T / 2) # N-vector
    
    return phi

