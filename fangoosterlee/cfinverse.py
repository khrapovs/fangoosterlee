#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inverse of characteristic function
==================================

Read Carr & Madan (1999) for idea of derivation

"""

import numpy as np

__all__ = ['cfinverse']


def cfinverse(psi, A=-1e2, B=1e2, N=1e5):
    """Discrete Fourier inverse

    Parameters
    ----------
    psi : function
        Characteristic function dependent only on u
    A : float, optional
        Lower limit of integration
    A : float, optional
        Upper limit of integration
    N : int, optional
        Number of discrete point for evaluation

    """
    eta = (N - 1) / N * 2 * np.pi / (B - A)
    lmbd = (B - A) / (N - 1)
    k = np.arange(N)
    x = A + lmbd * k
    v = eta * k
    y = psi(v) * np.exp(- 1j * A * v) * eta / np.pi
    f = np.fft.fft(y)
    return x, f.real
