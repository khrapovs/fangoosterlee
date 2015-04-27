#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inverse of characteristic function
==================================

Read Carr & Madan (1999) for idea of derivation

"""

import numpy as np

__all__ = ['cfinverse']


def cfinverse(psi, alim=-1e5, blim=1e5, points=1e5):
    """Discrete Fourier inverse.

    Inverts characteristic function to obtain the density.

    Parameters
    ----------
    psi : function
        Characteristic function dependent only on u
    alim : float, optional
        Lower limit of integration
    blim : float, optional
        Upper limit of integration
    points : int, optional
        Number of discrete points for evaluation

    Returns
    -------
    grid : (points, ) array
        Domain of the resulting density
    density : (points, ) array
        Density values

    """
    eta = (points - 1) / points * 2 * np.pi / (blim - alim)
    lmbd = (blim - alim) / (points - 1)
    karg = np.arange(points)
    grid = alim + lmbd * karg
    varg = eta * karg
    val = psi(varg) * np.exp(- 1j * alim * varg) * eta / np.pi
    density = np.fft.fft(val).real
    return grid, density
