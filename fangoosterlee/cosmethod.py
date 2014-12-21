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

    L, c1, c2, a, b = model.cos_restriction()

    if call:
        U = 2 / (b - a) * (xi(k,a,b,0,b) - psi(k,a,b,0,b)) # N-vector
    else:
        U = - 2 / (b - a) * (xi(k,a,b,a,0) - psi(k,a,b,a,0)) # N-vector

    CF = lambda x: model.charfun(x)

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
