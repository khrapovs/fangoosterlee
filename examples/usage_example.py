#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Usage examples of Fang-Oosterlee COS method
-------------------------------------------

"""

from fangoosterlee import (GBM, GBMParam, VarGamma, VarGammaParam,
                           Heston, HestonParam, ARG, cosmethod)

S, K = 100, 90
riskfree, maturity = 0, 30/365
sigma = .15

model = GBM(GBMParam(sigma=sigma), riskfree, maturity)
premium = cosmethod(model, S=S, K=K, T=maturity, r=riskfree, call=True)
print(premium)

nu = .2
theta = -.14
sigma = .25
param = VarGammaParam(theta=theta, nu=nu, sigma=sigma)
model = VarGamma(param, riskfree, maturity)
premium = cosmethod(model, S=S, K=K, T=maturity, r=riskfree, call=True)
print(premium)

lm = 1.5768
mu = .12**2
eta = .5751
rho = -.0
sigma = .12**2
param = HestonParam(lm=lm, mu=mu, eta=eta, rho=rho, sigma=sigma)
model = Heston(param, riskfree, maturity)
premium = cosmethod(model, S=S, K=K, T=maturity, r=riskfree, call=True)
print(premium)

rho = .55
delta = .75
mu = .2**2/365
sigma = .2**2/365
phi = -.0
theta1 = -16.0
theta2 = 20.95
model = ARG(rho, delta, mu, sigma, phi, theta1, theta2, riskfree, maturity)
premium = cosmethod(model, S=S, K=K, T=maturity, r=riskfree, call=True)
print(premium)
