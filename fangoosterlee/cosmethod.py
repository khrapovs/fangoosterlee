#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
COS method
==========

The method comes from [1]_

The original code is found at
http://www.wilmott.com/messageview.cfm?catid=34&threadid=78554

References
----------

.. [1] Fang, F., & Oosterlee, C. W. (2009).
    A Novel Pricing Method for European Options
    Based on Fourier-Cosine Series Expansions.
    *SIAM Journal on Scientific Computing*, 31(2), 826. doi:10.1137/080718061
    <http://ta.twi.tudelft.nl/mf/users/oosterle/oosterlee/COS.pdf>

Examples
--------

"""
from __future__ import division, print_function

import numpy as np

__all__ = ['cosmethod']


def cosmethod(model, price=100, strike=90, maturity=.1, riskfree=0, call=True):
    """COS method.

    Parameters
    ----------
    model : instance of specific model class
        The method depends on availability of two methods:
            - charfun
            - cos_restriction
    price : array_like
        Current asset price
    riskfree : array_like
        Risk-free rate, annualized
    maturity : float
        Fraction of a year

    Returns
    -------
    premium : array_like
        Option premium

    """
    if not hasattr(model, 'charfun'):
        raise Exception('Characteristic function is not available!')
    if not hasattr(model, 'cos_restriction'):
        raise Exception('COS restriction is not available!')

    N = 2**10

    # d-vector
    moneyness = np.log(price / strike)
    # N-vector
    k = np.arange(N)
    # N-vector
    unit = np.append(.5, np.ones(N-1))

    L, c1, c2, a, b = model.cos_restriction()

    if call:
        # N-vector
        U = 2 / (b - a) * (xi(k, a, b, 0, b) - psi(k, a, b, 0, b))
    else:
        # N-vector
        U = - 2 / (b - a) * (xi(k, a, b, a, 0) - psi(k, a, b, a, 0))

    # N-vector
    phi = model.charfun(k * np.pi / (b-a))

    # N x d arrays
    X1 = np.tile(phi[:, np.newaxis], (1, np.size(strike)))
    X2 = np.exp(1j * k[:, np.newaxis] * np.pi * (moneyness-a) / (b-a))
    X3 = np.tile(U[:, np.newaxis], (1, np.size(strike)))

    # d-vector
    ret = np.dot(unit, X1 * X2 * X3)

    # d-vector
    premium = strike * np.exp(- riskfree * maturity) * np.real(ret)

    return premium


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
