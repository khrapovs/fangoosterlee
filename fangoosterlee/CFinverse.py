#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:20:06 2012

@author: khrapov

Read Carr & Madan (1999) for idea of derivation

"""

import numpy as np

# A, B are limits of integration
# N number of iscrete point for evaluation
# psi is characteristic function dependent only on u

# Discrete Fourier inverse
def CFinverse(psi, A = -1e2, B = 1e2, N = 1e5):
    eta = (N - 1) / N * 2 * np.pi / (B - A)
    lmbd = (B - A) / (N - 1)
    #print eta, lmbd
    k = np.arange(0, N)
    x = A + lmbd * k
    v = eta * k
    y = psi(v) * np.exp(- 1j * A * v) * eta / np.pi
    f = np.fft.fft(y)
    return x, f.real
