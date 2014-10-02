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

from math import exp, log, sqrt, erf

def phi(x):
    return .5 * ( 1. + erf(x / sqrt(2)) )

## Black-Scholes Function
def BS(S, K, T, r, sig, cp):
    d1 = (log(S/K) + (r + sig**2/2)*T) / (sig*sqrt(T))
    d2 = d1 - sig*sqrt(T)
    if cp == 'C':
        value = S*phi(d1) - K*exp(-r*T)*phi(d2)
    if cp == 'P':
        value = K*exp(-r*T)*phi(-d2) - S*phi(-d1)
    return value

## Function to find BS Implied Vol using Bisection Method
def impvol(S, K, T, r, market, cp, tol = 1e-3, fcount = 1e3):
    sig, sig_u, sig_d = .2, 1., 1e-3
    count = 0
    err = BS(S, K, T, r, sig, cp) - market

    ## repeat until error is sufficiently small or counter hits 1000
    while abs(err) > tol and count < fcount:
        if err < 0:
            sig_d = sig
            sig = (sig_u + sig)/2
        else:
            sig_u = sig
            sig = (sig_d + sig)/2
        
        err = BS(S, K, T, r, sig, cp) - market
        count = count + 1
    
    ## return NA if counter hit 1000
    if count == 1e3:
        return 0
    else:
        return sig
