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


def cosmethod(model, moneyness=0., call=True):
    """COS method.

    Parameters
    ----------
    model : instance of specific model class
        The method depends on availability of two methods:
            - charfun
            - cos_restriction
    moneyness : array_like
        Moneyness of the option, np.log(strike/price) - riskfree * maturity
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
    kvec = np.arange(npoints)[:, np.newaxis] * np.pi
    # (npoints, ) array
    unit = np.append(.5, np.ones(npoints-1))

    alim, blim = model.cos_restriction()

    argc = (kvec, alim, blim, 0, blim)
    argp = (kvec, alim, blim, alim, 0)

    umat = 2 / (blim - alim) * (call * (xfun(*argc) - pfun(*argc))
        - np.logical_not(call) * (xfun(*argp) - pfun(*argp)))
    # (npoints, nobs) array
    pmat = model.charfun(kvec / (blim - alim))

    # (npoints, nobs) array
    xmat = np.exp(-1j * kvec * (moneyness + alim) / (blim - alim))

    # (nobs, ) array
    return np.exp(moneyness) * np.dot(unit, pmat * umat * xmat).real


def xfun(k, a, b, c, d):
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
    k = k/(b-a)
    return (np.cos(k * (d-a)) * np.exp(d) - np.cos(k * (c-a)) * np.exp(c) \
        + k * (np.sin(k * (d-a)) * np.exp(d) - np.sin(k * (c-a)) * np.exp(c)))\
        / (1 + k**2)


def pfun(k, a, b, c, d):
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
    k = k/(b-a)
    out = (np.sin(k[1:] * (d-a)) - np.sin(k[1:] * (c-a))) / k[1:]
    return np.vstack([(d - c) * np.ones_like(a), out])


if __name__ == '__main__':

    pass
