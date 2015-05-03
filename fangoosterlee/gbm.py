#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Geometric Brownian Motion
=========================

"""

import numpy as np

__all__ = ['GBM', 'GBMParam']


class GBMParam(object):

    """Parameter storage.

    """

    def __init__(self, sigma=.2):
        """Initialize class.

        Parameters
        ----------
        sigma : float

        """
        self.sigma = sigma


class GBM(object):

    """Geometric Brownian Motion.

    Attributes
    ----------
    param
        Model parameters

    Methods
    -------
    charfun
        Characteristic function
    cos_restriction
        Restrictions used in COS function

    """

    def __init__(self, param, riskfree, maturity):
        """Initialize the class.

        Parameters
        ----------
        param : GBMParam instance
            Model parameters
        riskfree : float
            Risk-free rate, annualized
        maturity : float
            Fraction of a year

        """
        self.param = param
        self.riskfree = riskfree
        self.maturity = maturity

    def charfun(self, arg):
        """Characteristic function.

        Parameters
        ----------
        arg : array_like
            Grid to evaluate the function

        Returns
        -------
        array_like
            Values of characteristic function

        """
        return np.exp(arg * self.riskfree * self.maturity * 1j
                      - arg**2 * self.param.sigma**2 * self.maturity / 2)

    def cos_restriction(self):
        """Restrictions used in COS function.

        Returns
        -------
        a : float
        b : float

        """
        # Truncation rate
        L = 10
        c1 = self.riskfree * self.maturity
        c2 = self.param.sigma**2 * self.maturity

        a = c1 - L * c2**.5
        b = c1 + L * c2**.5

        return a, b
