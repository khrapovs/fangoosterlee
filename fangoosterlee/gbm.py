#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Geometric Brownian Motion
=========================

"""

import numpy as np

__all__ = ['GBM']


class GBM(object):

    """Geometric Brownian Motion.

    Attributes
    ----------
    sigma
        Annualized volatility

    Methods
    -------
    charfun
        Characteristic function

    """

    def __init__(self, sigma, riskfree, maturity):
        """Initialize the class.

        Parameters
        ----------
        sigmma
            Annualized volatility

        """
        self.sigma = sigma
        self.riskfree = riskfree
        self.maturity = maturity

    def charfun(self, arg):
        """Characteristic function.

        Parameters
        ----------
        arg : array_like
            Grid to evaluate the function
        riskfree : float
            Risk-free rate, annualized
        maturity : float
            Fraction of a year

        Returns
        -------
        array_like
            Values of characteristic function

        """
        return np.exp(arg * self.riskfree * self.maturity * 1j
                      - arg**2 * self.sigma**2 * self.maturity / 2)

    def cos_restriction(self):

        # Truncation rate
        L = 100
        c1 = self.riskfree * self.maturity
        c2 = self.sigma**2 * self.maturity

        a = c1 - L * np.sqrt(c2)
        b = c1 + L * np.sqrt(c2)

        return L, c1, c2, a, b
