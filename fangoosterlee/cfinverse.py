#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inverse of characteristic function
==================================

Read Carr & Madan (1999) for idea of derivation

"""

import numpy as np

__all__ = ['cfinverse']


def cfinverse(psi, A=-1e5, B=1e5, points=1e5):
    """Discrete Fourier inverse.

    Inverts characteristic function to obtain the density.

    Parameters
    ----------
    psi : function
        Characteristic function dependent only on u
    A : float, optional
        Lower limit of integration
    B : float, optional
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
    eta = (points - 1) / points * 2 * np.pi / (B - A)
    lmbd = (B - A) / (points - 1)
    k = np.arange(points)
    grid = A + lmbd * k
    varg = eta * k
    y = psi(varg) * np.exp(- 1j * A * varg) * eta / np.pi
    density = np.fft.fft(y).real
    return grid, density
