#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 09:26:32 2012

@author: khrapov
"""
import numpy as np

# Variance Gamma characteristic function
def CFVG(u, P):
    # u is N-vector
    # T,r,sigma,nu,theta are scalars
    # returns N-vector
    T = P['T']
    r = P['r']
    theta = P['theta']
    nu = P['nu']
    sigma = P['sigma']
    
    omega = np.log(1 - theta * nu - sigma**2 * nu / 2) / nu # scalar
    phi = np.exp(u * (r + omega) * T * 1j) # N-vector
    phi = phi * ((1 - 1j * theta * nu * u + sigma**2 * nu * u**2 / 2) \
        ** (- T / nu))
    
    return phi

