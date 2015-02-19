#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Testing suite for COS method.

"""
from __future__ import print_function, division

import unittest as ut

from ..fangoosterlee import cosmethod
from ..examples import (GBM, GBMParam, VarGamma, VarGammaParam,
                        Heston, HestonParam)


class COSTestCase(ut.TestCase):
    """Test COS method."""

    def test_gbm(self):
        """Test GBM model."""

        price, strike = 100, 90
        riskfree, maturity = 0, 30/365

        sigma = .15

        model = GBM(GBMParam(sigma=sigma), riskfree, maturity)
        premium = cosmethod(model, price=price, strike=strike,
                            maturity=maturity, riskfree=riskfree, call=True)
        self.assertEqual(premium.shape, (1,))

    def test_vargamma(self):
        """Test VarGamma model."""

        price, strike = 100, 90
        riskfree, maturity = 0, 30/365

        nu = .2
        theta = -.14
        sigma = .25

        param = VarGammaParam(theta=theta, nu=nu, sigma=sigma)
        model = VarGamma(param, riskfree, maturity)
        premium = cosmethod(model, price=price, strike=strike,
                            maturity=maturity, riskfree=riskfree, call=True)

        self.assertEqual(premium.shape, (1,))

    def test_heston(self):
        """Test Heston model."""

        price, strike = 100, 90
        riskfree, maturity = 0, 30/365

        lm = 1.5768
        mu = .12**2
        eta = .5751
        rho = -.0
        sigma = .12**2

        param = HestonParam(lm=lm, mu=mu, eta=eta, rho=rho, sigma=sigma)
        model = Heston(param, riskfree, maturity)
        premium = cosmethod(model, price=price, strike=strike,
                            maturity=maturity, riskfree=riskfree, call=True)

        self.assertEqual(premium.shape, (1,))


if __name__ == '__main__':
    ut.main()
