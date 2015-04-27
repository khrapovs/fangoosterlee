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

"""
from __future__ import division, print_function

import numpy as np

__all__ = ['cosmethod']


def cosmethod(model, moneyness=0., maturity=.1, riskfree=0, call=True):
    """COS method.

    Parameters
    ----------
    model : instance of specific model class
        The method depends on availability of two methods:
            - charfun
            - cos_restriction
    moneyness : array_like
        Moneyness of the option, -np.log(strike/price)
    maturity : array_like
        Fraction of a year
    riskfree : array_like
        Risk-free rate, annualized
    call : bool array_like
        Call/Put flag

    Returns
    -------
    premium : array_like
        Option premium normalized by asset price

    Notes
    -----
    `charfun` method (risk-neutral conditional chracteristic function)
    of `model` instance should depend on
    one argument only (array_like) and should return
    array_like of the same dimension.

    `cos_restriction` method of `model` instance takes `maturity`
    and `riskfree` as array arguments,
    and returns five corresponding arrays (a, b).

    """
    if not hasattr(model, 'charfun'):
        raise Exception('Characteristic function is not available!')
    if not hasattr(model, 'cos_restriction'):
        raise Exception('COS restriction is not available!')

    npoints = 2**10
    # (npoints, 1) array
    kvec = np.arange(npoints)[:, np.newaxis]
    # (npoints, ) array
    unit = np.append(.5, np.ones(npoints-1))

    a, b = model.cos_restriction()

    umat = 2 / (b - a) * (call * (xi(kvec, a, b, 0, b) - psi(kvec, a, b, 0, b))
        - np.logical_not(call) * (xi(kvec, a, b, a, 0)
        - psi(kvec, a, b, a, 0)))
    # (npoints, nobs) array
    phi = model.charfun(kvec * np.pi / (b-a))

    # (npoints, nobs) array
    xmat = np.exp(1j * kvec * np.pi * (moneyness-a) / (b-a))

    # (nobs, ) array
    ret = np.dot(unit, phi * umat * xmat)

    # (nobs, ) array
    premium = np.exp(- moneyness - riskfree * maturity) * np.real(ret)

    return premium


def xi(k, a, b, c, d):
    """Xi function.

    Parameters
    ----------
    k : (n, 1) array
    a : float or (m, ) array
    b : float or (m, ) array
    c : float or (m, ) array
    d : float or (m, ) array

    Returns
    -------
    (n, m) array

    """
    # k is N-vector
    # a,b,c,d are scalars
    # returns N-vector
    return 1 / (1 + (k * np.pi / (b-a)) ** 2) \
        * (np.cos(k * np.pi * (d-a)/(b-a)) * np.exp(d) \
        - np.cos(k * np.pi * (c-a)/(b-a)) * np.exp(c) \
        + k * np.pi / (b-a) * np.sin(k * np.pi * (d-a)/(b-a)) * np.exp(d) \
        - k * np.pi / (b-a) * np.sin(k * np.pi * (c-a)/(b-a)) * np.exp(c))


def psi(k, a, b, c, d):
    """Psi function.

    Parameters
    ----------
    k : (n, 1) array
    a : float or (m, ) array
    b : float or (m, ) array
    c : float or (m, ) array
    d : float or (m, ) array

    Returns
    -------
    (n, m) array

    """
    # k is N-vector
    # a,b,c,d are scalars
    # returns N-vector
    if isinstance(a, float):
        size = 1
    else:
        size = a.size
    out = (np.sin(k[1:] * np.pi * (d-a)/(b-a)) \
        - np.sin(k[1:] * np.pi * (c-a)/(b-a))) * (b-a) / k[1:] / np.pi
    return np.vstack([(d - c) * np.ones(size), out])


if __name__ == '__main__':

    pass
