#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Variance Gamma process
======================

"""

import numpy as np

__all__ = ['VarGamma', 'VarGammaParam']


class VarGammaParam(object):

    """Parameter storage.

    """

    def __init__(self, theta=-.14, nu=.2, sigma=.25):
        """Initialize class.

        Parameters
        ----------
        nu : float
        theta : float
        sigma : float

        """
        self.sigma = sigma
        self.nu = nu
        self.theta = theta


class VarGamma(object):

    """Variance Gamma process.

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
        param : VarGammaParam instance
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
        theta, nu, sigma = self.param.theta, self.param.nu, self.param.sigma
        omega = np.log(1 - theta * nu - sigma**2 * nu/2) / nu
        phi = np.exp(arg * (self.riskfree + omega) * self.maturity * 1j)
        phi = phi * ((1 - 1j * theta * nu * arg \
            + sigma**2 * nu * arg**2 / 2) ** (- self.maturity / nu))

        return phi

    def cos_restriction(self):
        """Restrictions used in COS function.

        Returns
        -------
        a : float
        b : float

        """

        theta, nu, sigma = self.param.theta, self.param.nu, self.param.sigma
        L = 10
        c1 = (self.riskfree + theta) * self.maturity
        c2 = (sigma**2 + nu * theta**2) * self.maturity
        c4 = 3 * (sigma**4 * nu + 2 * theta**4 * nu**3 \
            + 4 * sigma**2 * theta**2 * nu**2) * self.maturity

        a = c1 - L * (c2 + c4**.5)**.5
        b = c1 + L * (c2 + c4**.5)**.5

        return a, b
