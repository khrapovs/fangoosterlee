#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Autoregressive Gamma Process
============================

"""

import numpy as np

__all__ = ['ARG', 'ARGParam']


class ARGParam(object):

    """Parameter storage.

    """

    def __init__(self, rho=.55, delta=.75, mu=.2**2/365,
                 sigma=.2**2/365, phi=-.1, theta1=-16, theta2=21):
        """Initialize class.

        Parameters
        ----------
        rho : float
        delta : float
        mu : float
        sigma : float
        phi : float
        theta1 : float
        theta2 : float

        """
        self.rho = rho
        self.delta = delta
        self.mu = mu
        self.sigma = sigma
        self.phi = phi
        self.theta1 = theta1
        self.theta2 = theta2


class ARG(object):

    """Autoregressive Gamma Process.

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
        param : ARGParam instance
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
        riskfree : float
            Risk-free rate, annualized
        maturity : float
            Fraction of a year

        Returns
        -------
        array_like
            Values of characteristic function

        """

        rho = self.param.rho
        delta = self.param.delta
        mu = self.param.mu
        sigma = self.param.sigma
        phi = self.param.phi
        theta1 = self.param.theta1
        theta2 = self.param.theta2

        scale = mu * (1 - rho) / delta
        betap = rho / scale
        n = int(self.maturity * 365)

        # distribution of one period volatility
        a = lambda u: rho * u / (1 + scale * u)
        b = lambda u: delta * np.log(1 + scale * u)

        center = phi / (scale * (1 + rho))**.5

        alpha = lambda v: (((theta2-.5) * (1-phi**2) + center) * v \
            - .5 * v**2 * (1 - phi**2) )

        # Risk-neutral parameters
        factor = 1 / (1 + scale * (theta1 + alpha(theta2)))
        scale_star = scale * factor
        betap_star = betap * factor
        rho_star = scale_star * betap_star

        a_star = lambda u: rho_star * u / (1 + scale_star * u)
        b_star = lambda u: delta * np.log(1 + scale_star * u)

        beta  = lambda v: v * a_star(- center)
        gamma = lambda v: v * b_star(- center)

        # joint distribution
        lf = lambda u, v: a(u + alpha(v)) + beta(v)
        gf = lambda u, v: b(u + alpha(v)) + gamma(v)

        # risk-neutral 1-period joint distribution
        lfQ = lambda u, v: lf(theta1 + u, theta2 + v) \
            - lf(theta1, theta2)
        gfQ = lambda u, v: gf(theta1 + u, theta2 + v) \
            - gf(theta1, theta2)
        # n-period cumulative return distribution
        psin = lambda n, v: lfQ(0, v) if n == 1 else lfQ(psin(n-1, v), v)
        upsn = lambda n, v: gfQ(0, v) if n == 1 else gfQ(psin(n-1, v), v) \
            + upsn(n-1, v)

        psi = np.exp( - psin(n, -1j * arg) * sigma - upsn(n, -1j * arg) )

        return psi

    def cos_restriction(self):
        """Restrictions used in COS function.

        Returns
        -------
        a : float
        b : float

        """
        L = 100
        c1 = self.riskfree * self.maturity
        c2 = self.param.mu * self.maturity * 365

        a = c1 - L * c2**.5
        b = c1 + L * c2**.5

        return a, b
