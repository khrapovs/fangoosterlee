#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Variance Gamma process
======================

"""

import numpy as np

__all__ = ['VarGamma']


class VarGamma(object):

    """Variance Gamma process.

    Attributes
    ----------
    sigma
        Annualized volatility

    Methods
    -------
    charfun
        Characteristic function

    """

    def __init__(self, theta, nu, sigma, riskfree, maturity):
        """Initialize the class.

        Parameters
        ----------
        sigmma
            Annualized volatility

        """
        self.theta = theta
        self.nu = nu
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
        omega = np.log(1 - self.theta * self.nu \
            - self.sigma**2 * self.nu / 2) / self.nu
        phi = np.exp(arg * (self.riskfree + omega) \
            * self.maturity * 1j)
        phi = phi * ((1 - 1j * self.theta * self.nu * arg \
            + self.sigma**2 * self.nu * arg**2 / 2) \
            ** (- self.maturity / self.nu))

        return phi

    def cos_restriction(self):

        # Truncation rate
        # Truncation rate
        L = 10 # scalar
        c1 = (self.riskfree + self.theta) * self.maturity
        c2 = (self.sigma**2 + self.nu * self.theta**2) * self.maturity
        c4 = 3 * (self.sigma**4 * self.nu + 2 * self.theta**4 * self.nu**3 \
            + 4 * self.sigma**2 * self.theta**2 * self.nu**2) * self.maturity

        a = c1 - L * np.sqrt(c2 + np.sqrt(c4)) # scalar
        b = c1 + L * np.sqrt(c2 + np.sqrt(c4)) # scalar

        return L, c1, c2, a, b
