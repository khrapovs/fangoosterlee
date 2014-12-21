#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
COS method
==========

The method comes from Fang & Oosterlee (2009)

The code is found at
http://www.wilmott.com/messageview.cfm?catid=34&threadid=78554

Working example
---------------
d = 1e1
S = 100
K = np.linspace(80, 120, d) # d-vector
T = .1
r = .1
sigma = .12
nu = .2
theta = -.14

"""
from __future__ import division, print_function

import numpy as np
from .gbm import GBM
#from CFHeston import CFHeston
#from CFVG import CFVG
#from CFARG import CFARG

__all__ = ['cosmethod']


def cosmethod(model, S=100, K=90, T=.1, r=0, call=True):
    """COS method.


    """
    N = 2**10

    # d-vector
    x = np.log(S / K)
    # N-vector
    k = np.arange(N)
    unit = np.append(.5, np.ones(N-1)) # N-vector

    distr = 'GBM'

    if distr == 'GBM':
        #P = {'T': T, 'r': r, 'sigma': .12}
        sigma = model.sigma

        L, c1, c2, a, b = model.cos_restriction()

        if call:
            U = 2 / (b - a) * (xi(k,a,b,0,b) - psi(k,a,b,0,b)) # N-vector
        else:
            U = - 2 / (b - a) * (xi(k,a,b,a,0) - psi(k,a,b,a,0)) # N-vector

        CF = lambda x: model.charfun(x)

    elif distr == 'VG':
        #P = {'T': T, 'r': r, 'nu': .2, 'theta': -.14, 'sigma': .12}
        nu = P['nu']
        theta = P['theta']
        sigma = P['sigma']

        # Truncation rate
        L = 10 # scalar
        c1 = (r + theta) * T # scalar
        c2 = (sigma**2 + nu * theta**2) * T # scalar
        c4 = 3 * (sigma**4 * nu + 2 * theta**4 * nu**3 \
            + 4 * sigma**2 * theta**2 * nu**2) * T # scalar

        a = c1 - L * np.sqrt(c2 + np.sqrt(c4)) # scalar
        b = c1 + L * np.sqrt(c2 + np.sqrt(c4)) # scalar

        if P['cp_flag'] == 'C':
            U = 2 / (b - a) * (xi(k,a,b,0,b) - psi(k,a,b,0,b)) # N-vector
        else:
            U = - 2 / (b - a) * (xi(k,a,b,a,0) - psi(k,a,b,a,0)) # N-vector

        CF = lambda x: model.charfun(x)

    elif distr == 'Heston':
        #P = {'T': T, 'r': r, 'lm': 1.5768, 'meanV': .0398, 'eta': .5751, \
        #    'rho': -.5711, 'v0': .0175}
        lm = P['lm']
        meanV = P['meanV']
        eta = P['eta']
        rho = P['rho']
        v0 = P['v0']

        # Truncation for Heston:
        L = 12 # scalar
        c1 = r * T + (1 - np.exp(-lm * T)) \
            * (meanV - v0) / 2 / lm - .5 * meanV * T # scalar

        c2 = 1/(8 * lm**3) * (eta * T * lm * np.exp(-lm * T) \
            * (v0 - meanV) * (8 * lm * rho - 4 * eta) \
            + lm * rho * eta * (1 - np.exp(-lm * T)) \
            * (16 * meanV - 8 * v0) + 2 * meanV * lm * T \
            * (-4 * lm * rho * eta + eta**2 + 4 * lm**2) \
            + eta**2 * ((meanV - 2 * v0) * np.exp(-2*lm*T) \
            + meanV * (6 * np.exp(-lm * T) - 7) + 2 * v0) \
            + 8 * lm**2 * (v0 - meanV) * (1 - np.exp(-lm*T))) # scalar

        a = c1 - L * np.sqrt(np.abs(c2)) # scalar
        b = c1 + L * np.sqrt(np.abs(c2)) # scalar

        if P['cp_flag'] == 'C':
            U = 2 / (b - a) * (xi(k,a,b,0,b) - psi(k,a,b,0,b)) # N-vector
        else:
            U = - 2 / (b - a) * (xi(k,a,b,a,0) - psi(k,a,b,a,0)) # N-vector

        CF = CFHeston

    elif distr == 'ARG':
        #P = {'T': T, 'r': r, 'rho': .9, 'delta': 1.1, 'dailymean': .3**2, \
        #    'theta1': 0, 'theta2': 0, 'phi': .0}

        # Truncation rate
        L = 100 # scalar
        c1 = r * T # scalar
        c2 = P['dailymean'] * T * 365 # scalar

        a = c1 - L * np.sqrt(c2) # scalar
        b = c1 + L * np.sqrt(c2) # scalar

        if P['cp_flag'] == 'C':
            U = 2 / (b - a) * (xi(k,a,b,0,b) - psi(k,a,b,0,b)) # N-vector
        else:
            U = - 2 / (b - a) * (xi(k,a,b,a,0) - psi(k,a,b,a,0)) # N-vector

        CF = CFARG

    #phi = CF(k * np.pi / (b-a), P) # N-vector
    phi = CF(k * np.pi / (b-a)) # N-vector

    X1 = np.tile(phi[:,np.newaxis], (1, np.size(K))) # N x d matrix
    X2 = np.exp(1j * k[:, np.newaxis] * np.pi * (x-a) / (b-a)) # N x d matrix
    X3 = np.tile(U[:, np.newaxis], (1, np.size(K))) # N x d matrix

    ret = np.dot(unit, X1 * X2 * X3) # d-vector

    price = K * np.exp(- r * T) * np.real(ret) # d-vector

    return price

def xi(k,a,b,c,d):
    # k is N-vector
    # a,b,c,d are scalars
    # returns N-vector
    ret = 1 / (1 + (k * np.pi / (b-a)) ** 2) \
        * (np.cos(k * np.pi * (d-a)/(b-a)) * np.exp(d) \
        - np.cos(k * np.pi * (c-a)/(b-a)) * np.exp(c) \
        + k * np.pi / (b-a) * np.sin(k * np.pi * (d-a)/(b-a)) * np.exp(d) \
        - k * np.pi / (b-a) * np.sin(k * np.pi * (c-a)/(b-a)) * np.exp(c))
    return ret

def psi(k,a,b,c,d):
    # k is N-vector
    # a,b,c,d are scalars
    # returns N-vector
    ret = (np.sin(k[1:] * np.pi * (d-a)/(b-a)) \
        - np.sin(k[1:] * np.pi * (c-a)/(b-a))) * (b-a) / k[1:] / np.pi
    ret = np.append(d - c, ret)
    return ret


if __name__ == '__main__':

    pass
