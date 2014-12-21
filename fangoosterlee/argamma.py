#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Autoregressive Gamma Process
============================

"""

import numpy as np

__all__ = ['ARG']


class ARG(object):

    """Autoregressive Gamma Process.

    Attributes
    ----------
    sigma
        Annualized volatility

    Methods
    -------
    charfun
        Characteristic function

    """

    def __init__(self, rho, delta, mu, sigma, phi, theta1, theta2,
                 riskfree, maturity):
        """Initialize the class.

        Parameters
        ----------
        sigmma
            Annualized volatility

        """
        self.rho = rho
        self.delta = delta
        self.mu = mu
        self.sigma = sigma
        self.phi = phi
        self.theta1 = theta1
        self.theta2 = theta2
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
        scale = self.mu * (1 - self.rho) / self.delta
        betap = self.rho / scale
        n = int(self.maturity * 365)

        # distribution of one period volatility
        a = lambda u: self.rho * u / (1 + scale * u)
        b = lambda u: self.delta * np.log(1 + scale * u)

        center = self.phi / (scale * (1 + self.rho))**.5

        alpha = lambda v: (((self.theta2-.5) * (1-self.phi**2) + center) * v \
            - .5 * v**2 * (1 - self.phi**2) )

        # Risk-neutral parameters
        factor = 1 / (1 + scale * (self.theta1 + alpha(self.theta2)))
        scale_star = scale * factor
        betap_star = betap * scale_star / scale
        rho_star = scale_star * betap_star

        a_star = lambda u: rho_star * u / (1 + scale_star * u)
        b_star = lambda u: self.delta * np.log(1 + scale_star * u)

        beta  = lambda v: v * a_star(- center)
        gamma = lambda v: v * b_star(- center)

        # joint distribution
        lf = lambda u, v: a(u + alpha(v)) + beta(v)
        gf = lambda u, v: b(u + alpha(v)) + gamma(v)

        # risk-neutral 1-period joint distribution
        lfQ = lambda u, v: lf(self.theta1 + u, self.theta2 + v) \
            - lf(self.theta1, self.theta2)
        gfQ = lambda u, v: gf(self.theta1 + u, self.theta2 + v) \
            - gf(self.theta1, self.theta2)
        # n-period cumulative return distribution
        psin = lambda n, v: lfQ(0, v) if n == 1 else lfQ(psin(n-1, v), v)
        upsn = lambda n, v: gfQ(0, v) if n == 1 else gfQ(psin(n-1, v), v) \
            + upsn(n-1, v)

        psi = np.exp( - psin(n, -1j * arg) * self.sigma - upsn(n, -1j * arg) )
        psi = psi * np.exp(- 1j * arg * self.riskfree * self.maturity)
        return psi

    def cos_restriction(self):

        # Truncation rate
        L = 100 # scalar
        c1 = self.riskfree * self.maturity # scalar
        c2 = self.mu * self.maturity * 365

        a = c1 - L * np.sqrt(c2)
        b = c1 + L * np.sqrt(c2)

        return L, c1, c2, a, b
