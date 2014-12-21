#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Usage examples of Fang-Oosterlee COS method
-------------------------------------------

"""

from fangoosterlee import GBM, VarGamma, cosmethod

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
