#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Usage examples of Fang-Oosterlee COS method
-------------------------------------------

"""

from fangoosterlee import GBM, VarGamma, Heston, cosmethod

sigma, riskfree, maturity = .15, 0, 30/365

model = GBM(sigma, riskfree, maturity)
premium = cosmethod(model, S=100, K=90, T=.1, r=riskfree, call=True)
print(premium)

nu = .2
theta = -.14
sigma = .25
model = VarGamma(theta, nu, sigma, riskfree, maturity)
premium = cosmethod(model, S=100, K=90, T=.1, r=riskfree, call=True)
print(premium)

lm = 1.5768
mu = .12**2
eta = .5751
rho = -.0
sigma = .12**2
model = Heston(lm, mu, eta, rho, sigma, riskfree, maturity)
premium = cosmethod(model, S=100, K=90, T=.1, r=riskfree, call=True)
print(premium)
