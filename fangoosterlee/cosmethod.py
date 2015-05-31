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
import numexpr as ne

__all__ = ['cosmethod']


def cosmethod(model, moneyness=0., call=True, npoints=2**10):
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
    npoints : int
        Number of points on the grid. The more the better, but slower.

    Returns
    -------
    array_like
        Option premium normalized by asset price

    Notes
    -----
    `charfun` method (risk-neutral conditional chracteristic function)
    of `model` instance should depend on
    one argument only (array_like) and should return
    array_like of the same dimension.

    `cos_restriction` method of `model` instance takes `maturity`
    and `riskfree` as array arguments,
    and returns two corresponding arrays (a, b).

    """
    if not hasattr(model, 'charfun'):
        raise Exception('Characteristic function is not available!')
    if not hasattr(model, 'cos_restriction'):
        raise Exception('COS restriction is not available!')

    # (nobs, ) arrays
    alim, blim = model.cos_restriction()
    # (npoints, nobs) array
    kvec = np.arange(npoints)[:, np.newaxis] * np.pi / (blim - alim)
    # (npoints, ) array
    unit = np.append(.5, np.ones(npoints-1))
    # Arguments
    argc = (kvec, alim, blim, 0, blim)
    argp = (kvec, alim, blim, alim, 0)
    # (nobs, ) array
    put = np.logical_not(call)
    # (npoints, nobs) array
    umat = 2 / (blim - alim) * (call * xfun(*argc) - put * xfun(*argp))
    # (npoints, nobs) array
    pmat = model.charfun(kvec)
    # (npoints, nobs) array
    xmat = np.exp(-1j * kvec * (moneyness + alim))
    # (nobs, ) array
    return np.exp(moneyness) * np.dot(unit, pmat * umat * xmat).real


def xfun(k, a, b, c, d):
    """Xi-Psi function.

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
#    out0 = (np.cos(k * (d-a)) * np.exp(d) - np.cos(k * (c-a)) * np.exp(c)
#        + k * (np.sin(k * (d-a)) * np.exp(d) - np.sin(k * (c-a)) * np.exp(c)))\
#        / (1 + k**2)
#    out1 = (np.sin(k[1:] * (d-a)) - np.sin(k[1:] * (c-a))) / k[1:]

    out0 = ne.evaluate(("(cos(k * (d-a)) * exp(d) - cos(k * (c-a)) * exp(c)"
        "+ k * (sin(k * (d-a)) * exp(d) - sin(k * (c-a)) * exp(c)))"
        "/ (1 + k**2)"))
    k1 = k[1:]
    out1 = ne.evaluate("(sin(k1 * (d-a)) - sin(k1 * (c-a))) / k1")

    out1 = np.vstack([(d - c) * np.ones_like(a), out1])

    return out0 - out1


if __name__ == '__main__':

    pass
