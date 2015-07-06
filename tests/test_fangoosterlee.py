#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Testing suite for COS method.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np
import scipy.stats as scs

from impvol import impvol_bisection, blackscholes_norm, lfmoneyness
from fangoosterlee import (cosmethod, cfinverse, GBM, GBMParam,
                           VarGamma, VarGammaParam,
                           Heston, HestonParam, ARG, ARGParam)


class COSTestCase(ut.TestCase):
    """Test COS method."""

    def test_gbm(self):
        """Test GBM model."""

        price, strike = 100, 110
        riskfree, maturity = .01, 30/365
        call = True
        put = np.logical_not(call)
        moneyness = lfmoneyness(price, strike, riskfree, maturity)

        sigma = .15

        model = GBM(GBMParam(sigma=sigma), riskfree, maturity)
        premium = cosmethod(model, moneyness=moneyness, call=call)
        premium_true = blackscholes_norm(moneyness, maturity, sigma, call)
        impvol_model = impvol_bisection(moneyness, maturity, premium, call)

        self.assertEqual(premium.shape, (1,))
        np.testing.assert_array_almost_equal(premium, premium_true, 3)
        np.testing.assert_array_almost_equal(impvol_model, sigma, 2)

        moneyness = np.linspace(0, .1, 10)
        premium = cosmethod(model, moneyness=moneyness, call=call)
        premium_true = blackscholes_norm(moneyness, maturity, sigma, call)
        impvol_model = impvol_bisection(moneyness, maturity, premium, call)
        impvol_true = np.ones_like(impvol_model) * sigma

        self.assertEqual(premium.shape, moneyness.shape)
        np.testing.assert_array_almost_equal(premium, premium_true, 2)
        np.testing.assert_array_almost_equal(impvol_model, impvol_true, 2)

        riskfree = np.zeros_like(moneyness)
        premium = cosmethod(model, moneyness=moneyness, call=call)
        premium_true = blackscholes_norm(moneyness, maturity, sigma, call)
        impvol_model = impvol_bisection(moneyness, maturity, premium, call)

        self.assertEqual(premium.shape, moneyness.shape)
        np.testing.assert_array_almost_equal(premium, premium_true, 3)
        np.testing.assert_array_almost_equal(impvol_model, sigma, 2)

        moneyness = np.linspace(-.1, 0, 10)
        premium = cosmethod(model, moneyness=moneyness, call=put)
        premium_true = blackscholes_norm(moneyness, maturity, sigma, put)
        impvol_model = impvol_bisection(moneyness, maturity, premium, put)

        np.testing.assert_array_almost_equal(premium, premium_true, 2)
        np.testing.assert_array_almost_equal(impvol_model, impvol_true, 2)

    def test_vargamma(self):
        """Test VarGamma model."""

        price, strike = 100, 90
        riskfree, maturity = 0, 30/365
        moneyness = np.log(strike/price) - riskfree * maturity

        nu = .2
        theta = -.14
        sigma = .25

        param = VarGammaParam(theta=theta, nu=nu, sigma=sigma)
        model = VarGamma(param, riskfree, maturity)
        premium = cosmethod(model, moneyness=moneyness, call=True)

        self.assertEqual(premium.shape, (1,))

        moneyness = np.linspace(-.1, .1, 10)
        premium = cosmethod(model, moneyness=moneyness, call=True)

        self.assertEqual(premium.shape, moneyness.shape)

        riskfree = np.zeros_like(moneyness)
        premium = cosmethod(model, moneyness=moneyness, call=True)

        self.assertEqual(premium.shape, moneyness.shape)


    def test_heston(self):
        """Test Heston model."""

        price, strike = 100, 90
        riskfree, maturity = 0, 30/365
        moneyness = np.log(strike/price) - riskfree * maturity

        lm = 1.5768
        mu = .12**2
        eta = .5751
        rho = -.0
        sigma = .12**2

        param = HestonParam(lm=lm, mu=mu, eta=eta, rho=rho, sigma=sigma)
        model = Heston(param, riskfree, maturity)
        premium = cosmethod(model, moneyness=moneyness, call=True)

        self.assertEqual(premium.shape, (1,))

        moneyness = np.linspace(-.1, .1, 10)
        premium = cosmethod(model, moneyness=moneyness, call=True)

        self.assertEqual(premium.shape, moneyness.shape)

        riskfree = np.zeros_like(moneyness)
        premium = cosmethod(model, moneyness=moneyness, call=True)

        self.assertEqual(premium.shape, moneyness.shape)

    def test_argamma(self):
        """Test ARG model."""

        price, strike = 100, 90
        riskfree, maturity = 0, 30/365
        moneyness = np.log(strike/price) - riskfree * maturity

        rho = .55
        delta = .75
        mu = .2**2/365
        sigma = .2**2/365
        phi = -.0
        theta1 = -16.0
        theta2 = 20.95

        param = ARGParam(rho=rho, delta=delta, mu=mu, sigma=sigma,
                     phi=phi, theta1=theta1, theta2=theta2)
        model = ARG(param, riskfree, maturity)
        premium = cosmethod(model, moneyness=moneyness, call=True)

        self.assertEqual(premium.shape, (1,))

        moneyness = np.linspace(-.1, .1, 10)
        premium = cosmethod(model, moneyness=moneyness, call=True)

        self.assertEqual(premium.shape, moneyness.shape)

        riskfree = np.zeros_like(moneyness)
        premium = cosmethod(model, moneyness=moneyness, call=True)

        self.assertEqual(premium.shape, moneyness.shape)

    def test_cfinverse(self):
        """Test Fourier inversion."""

        riskfree, maturity = 0, 30/365
        sigma = .15
        points = int(1e5)

        model = GBM(GBMParam(sigma=sigma), riskfree, maturity)

        grid, density = cfinverse(model.charfun, points=points,
                                  alim=-1e5, blim=1e5)

        loc = (riskfree - sigma**2/2) * maturity
        scale = sigma**2 * maturity
        norm_density = scs.norm.pdf(grid, loc=loc, scale=scale**.5)

        self.assertEqual(grid.shape, (points,))
        self.assertEqual(density.shape, (points,))

        good = np.abs(grid) < 2

        np.testing.assert_array_almost_equal(density[good], norm_density[good],
                                             decimal=2)


if __name__ == '__main__':
    ut.main()
