# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 09:25:19 2012

@author: khrapov
"""
import numpy as np

# Heston characteristic function
def CFHeston(u, P):
    # u is N-vector
    # T,r,lm,meanV,eta,rho,v0 are scalars
    # returns N-vector
    T = P['T']
    r = P['r']
    lm = P['lm']
    meanV = P['meanV']
    eta = P['eta']
    rho = P['rho']
    v0 = P['v0']
    
    d = np.sqrt((lm - 1j * rho * eta * u)**2 + (u**2 + 1j * u) * eta**2)
    g = (lm - 1j * rho * eta * u - d) / (lm - 1j * rho * eta * u + d)
    
    phi = np.exp(1j * u * r * T + v0 / eta**2 * (1 - np.exp(-d * T)) \
        / (1 - g * np.exp(-d * T)) * (lm - 1j * rho * eta * u - d))

    phi = phi * np.exp(lm * meanV / eta**2 * \
        (T * (lm - 1j * rho * eta * u - d) \
        - 2 * np.log((1-g * np.exp(-d * T)) / (1 - g))))
    
    return phi