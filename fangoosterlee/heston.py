#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Geometric Brownian Motion
=========================

"""

import numpy as np

__all__ = ['Heston']


class Heston(object):

    """Heston Stochastic Volatility model.

    Attributes
    ----------
    sigma
        Annualized volatility

    Methods
    -------
    charfun
        Characteristic function

    """

    def __init__(self, lm, mu, eta, rho, sigma, riskfree, maturity):
        """Initialize the class.

        Parameters
        ----------
        sigmma
            Annualized volatility

        """
        self.lm = lm
        self.mu = mu
        self.eta = eta
        self.rho = rho
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
        d = np.sqrt((self.lm - 1j * self.rho * self.eta * arg)**2 \
            + (arg**2 + 1j * arg) * self.eta**2)
        g = (self.lm - 1j * self.rho * self.eta * arg - d) \
            / (self.lm - 1j * self.rho * self.eta * arg + d)

        phi = np.exp(1j * arg * self.riskfree * self.maturity + self.sigma / self.eta**2 \
            * (1 - np.exp(-d * self.maturity)) \
            / (1 - g * np.exp(-d * self.maturity)) \
            * (self.lm - 1j * self.rho * self.eta * arg - d))

        phi = phi * np.exp(self.lm * self.mu / self.eta**2 * \
            (self.maturity * (self.lm - 1j * self.rho * self.eta * arg - d) \
            - 2 * np.log((1-g * np.exp(-d * self.maturity)) / (1 - g))))

        return phi

    def cos_restriction(self):

        # Truncation for Heston:
        L = 12
        c1 = self.riskfree * self.maturity + (1 - np.exp(-self.lm * self.maturity)) \
            * (self.mu - self.sigma) / 2 / self.lm - .5 * self.mu * self.maturity

        c2 = 1/(8 * self.lm**3) * (self.eta * self.maturity * self.lm * np.exp(-self.lm * self.maturity) \
            * (self.sigma - self.mu) * (8 * self.lm * self.rho - 4 * self.eta) \
            + self.lm * self.rho * self.eta * (1 - np.exp(-self.lm * self.maturity)) \
            * (16 * self.mu - 8 * self.sigma) + 2 * self.mu * self.lm * self.maturity \
            * (-4 * self.lm * self.rho * self.eta + self.eta**2 + 4 * self.lm**2) \
            + self.eta**2 * ((self.mu - 2 * self.sigma) * np.exp(-2*self.lm*self.maturity) \
            + self.mu * (6 * np.exp(-self.lm * self.maturity) - 7) + 2 * self.sigma) \
            + 8 * self.lm**2 * (self.sigma - self.mu) * (1 - np.exp(-self.lm*self.maturity)))

        a = c1 - L * np.sqrt(np.abs(c2)) # scalar
        b = c1 + L * np.sqrt(np.abs(c2)) # scalar

        return L, c1, c2, a, b



#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 09:25:19 2012

@author: khrapov
"""
import numpy as np

# Heston characteristic function
def CFHeston(u, P):
    # u is N-vector
    # T,r,lm,meanV,eta,rho,v0 are scalars
    # returns N-vector
    T = P['T']
    r = P['r']
    lm = P['lm']
    meanV = P['meanV']
    eta = P['eta']
    rho = P['rho']
    v0 = P['v0']

    d = np.sqrt((lm - 1j * rho * eta * u)**2 + (u**2 + 1j * u) * eta**2)
    g = (lm - 1j * rho * eta * u - d) / (lm - 1j * rho * eta * u + d)

    phi = np.exp(1j * u * r * T + v0 / eta**2 * (1 - np.exp(-d * T)) \
        / (1 - g * np.exp(-d * T)) * (lm - 1j * rho * eta * u - d))

    phi = phi * np.exp(lm * meanV / eta**2 * \
        (T * (lm - 1j * rho * eta * u - d) \
        - 2 * np.log((1-g * np.exp(-d * T)) / (1 - g))))

    return phi