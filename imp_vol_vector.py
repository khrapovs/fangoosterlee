# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:14:37 2013

@author: skhrapov
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 17:06:38 2012

@author: khrapovs
"""

# Test code:
# S - stock price
# K - strike
# T - maturity in years
# r - risk free rate annualized
# market - option price
#
#S, K, T, r, market, cp = 1, 1, 30./365, 0, .1, 'C'
#v = impvol(S, K, T, r, market, cp)
#print v

from math import sqrt
import numpy as np

def phi(x):
    return .5 * ( 1. + verf(x / sqrt(2)) )

## Black-Scholes Function
def BS(S, K, T, r, sig, cp):
    d1 = (np.log(S/K) + (r + sig**2/2)*T) / (sig*np.sqrt(T)) # N-vector
    d2 = d1 - sig * np.sqrt(T) # N-vector
    
    vc = S*phi(d1) - K*np.exp(-r*T)*phi(d2)
    vp = K*np.exp(-r*T)*phi(-d2) - S*phi(-d1)
    value = vc
    value[cp == 'P'] = vp[cp == 'P']
    return value  # N-vector

## Function to find BS Implied Vol using Bisection Method
def impvol(S, K, T, r, P, cp, tol = 1e-4, fcount = 1e3):
    # Convert to 1d arrays    
    S = np.atleast_1d(np.array(S))
    K = np.atleast_1d(np.array(K))
    T = np.atleast_1d(np.array(T))
    r = np.atleast_1d(np.array(r))
    P = np.atleast_1d(np.array(P))
    cp = np.atleast_1d(np.array(cp))
    
    n = len(K)
    sig, sig_u, sig_d = .2 * np.ones(n), 1. * np.ones(n), 1e-4 * np.ones(n)
    count = 0
    err = BS(S, K, T, r, sig, cp) - P # N-vector
    
    ## repeat until error is sufficiently small or counter hits 1000
    while (np.abs(err) > tol).any() and count < fcount:
        sig_d[err < 0] = sig[err < 0]
        sig[err < 0] = (sig_u[err < 0] + sig[err < 0])/2
        sig_u[err >= 0] = sig[err >= 0]
        sig[err >= 0] = (sig_d[err >= 0] + sig[err >= 0])/2
    
        err = BS(S, K, T, r, sig, cp) - P
        
        count = count + 1
    
#    print 'Error: ', err
#    print 'Count: ', count
    ## return NA if counter hit 1000
#    if count == fcount:
#        return 0
#    else:
#        return sig
    return sig

def verf(x):
    # save the sign of x
    #sign = 1 if x >= 0 else -1
    sign = np.sign(x)
    x = abs(x)

    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1) * t * np.exp(-x*x)

    return sign*y # erf(-x) = -erf(x)

def test():
    S = 1.
    K = np.array([1., .9])
    T = np.array([60., 30.])/365
    r = np.array([.0, .0001])
    P = np.array([.01, .02])
    cp = np.array(['C','P'])
    
    v = impvol(S, K, T, r, P, cp)
    
    print v

def test2():
    S = 1.
    K = 1.
    T = 30./365
    r = 0.
    P = .05
    cp = 'C'
    
    v = impvol(S, K, T, r, P, cp)
    
    print v

#test2()