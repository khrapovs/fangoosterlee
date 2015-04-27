#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Heston Stochastic Volatility model
==================================

"""

import numpy as np

__all__ = ['Heston', 'HestonParam']


class HestonParam(object):

    """Parameter storage.

    """

    def __init__(self, lm=1.5, mu=.12**2, eta=.57, rho=-.2, sigma=.12**2):
        """Initialize class.

        Parameters
        ----------
        lm : float
        mu : float
        eta : float
        rho : float
        sigma : float

        """
        self.lm = lm
        self.mu = mu
        self.eta = eta
        self.rho = rho
        self.sigma = sigma


class Heston(object):

    """Heston Stochastic Volatility model.

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
        param : HestonParam instance
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
        lm, mu, eta = self.param.lm, self.param.mu, self.param.eta
        rho, sigma = self.param.rho, self.param.sigma

        d = np.sqrt((lm - 1j * rho*eta*arg)**2 + (arg**2 + 1j*arg) * eta**2)
        g = (lm - 1j * rho * eta * arg - d) / (lm - 1j * rho * eta * arg + d)

        phi = np.exp(1j * arg * self.riskfree * self.maturity + sigma/eta**2 \
            * (1 - np.exp(-d * self.maturity)) \
            / (1 - g * np.exp(-d * self.maturity)) \
            * (lm - 1j * rho * eta * arg - d))

        phi = phi * np.exp(lm * mu / eta**2 * \
            (self.maturity * (lm - 1j * rho * eta * arg - d) \
            - 2 * np.log((1-g * np.exp(-d * self.maturity)) / (1 - g))))

        return phi

    def cos_restriction(self):
        """Restrictions used in COS function.

        Returns
        -------
        a : float
        b : float

        """

        lm, mu, eta = self.param.lm, self.param.mu, self.param.eta
        rho, sigma = self.param.rho, self.param.sigma

        L = 12
        c1 = self.riskfree * self.maturity \
            + (1 - np.exp(-lm * self.maturity)) \
            * (mu - sigma)/2/lm - mu * self.maturity / 2

        c2 = 1/(8 * lm**3) \
            * (eta * self.maturity * lm * np.exp(-lm * self.maturity) \
            * (sigma - mu) * (8 * lm * rho - 4 * eta) \
            + lm * rho * eta * (1 - np.exp(-lm * self.maturity)) \
            * (16 * mu - 8 * sigma) + 2 * mu * lm * self.maturity \
            * (-4 * lm * rho * eta + eta**2 + 4 * lm**2) \
            + eta**2 * ((mu - 2 * sigma) * np.exp(-2*lm*self.maturity) \
            + mu * (6 * np.exp(-lm*self.maturity) - 7) + 2 * sigma) \
            + 8 * lm**2 * (sigma - mu) * (1 - np.exp(-lm*self.maturity)))

        a = c1 - L * np.abs(c2)**.5
        b = c1 + L * np.abs(c2)**.5

        return a, b
