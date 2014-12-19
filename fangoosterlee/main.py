#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 11:15:02 2012

@author: khrapov
"""

import numpy as np
import matplotlib.pyplot as plt
from COS import COS
from CFHeston import CFHeston
from CFVG import CFVG
from CFGBM import CFGBM
from CFARG import CFARG
from CFinverse import CFinverse
from imp_vol import impvol

S = 1.
T = 30./365
r = .0

d = 1e2
K = np.exp(np.linspace(-.1, .1, d))

#distr = 'GBM'
distr = 'Heston'
#distr = 'VG'
#distr = 'ARG'

if distr == 'GBM':
    
    P = {'T': T,
         'r': r,
         'sigma': .25}
    psi = lambda u: CFGBM(u, P)

elif distr == 'VG':

    P = {'T': T,
         'r': r,
         'nu': .2,
         'theta': -.14,
         'sigma': .25}
    psi = lambda u: CFVG(u, P)

elif distr == 'Heston':

    #P = {'T': T, 'r': r, 'lm': 1.5768, 'meanV': .0398, 'eta': .5751, \
    #    'rho': -.5711, 'v0': .0175}
    P = {'T': T,
         'r': r,
         'lm': 1.5768,
         'meanV': .12**2,
         'eta': .5751,
         'rho': -.0,
         'v0': .12**2,
         'cp_flag' : 'C'}
    psi = lambda u: CFHeston(u, P)

elif distr == 'ARG':

    #P = {'T': T, 'r': r, 'rho': .9, 'delta': 1.1, 'dailymean': .3**2, \
    #    'theta1': 0, 'theta2': 0, 'phi': -.3}
    P = {'T': T,
         'r': r,
         'rho': .55,
         'delta': .75,
         'dailymean': .2**2/365,
         'v0': .2**2/365,
         'theta1': -16.0,
         'theta2': 20.95,
         'phi': -.0,
         'cp_flag' : 'C'}
    psi = lambda u: CFARG(u, P)


# Recover density of future cumulative return
B = 1e2
A = -B
N = 1e5

x_s, f_s = CFinverse(psi, A, B, N)

plt.plot(x_s, f_s)
plt.grid(1)
plt.xlim(-.2, .2)
plt.show()

# Plot characteristic function
u = np.linspace(-1e2, 1e2, 1e2)
plt.subplot(2,1,1)
plt.plot(u, np.real(psi(u)))
plt.grid(1)
plt.subplot(2,1,2)
plt.plot(u, np.imag(psi(u)))
plt.grid(1)
plt.show()

# Price option according to the model
Pr = COS(S, K, T, r, distr, P)
# compute implied volatilities
V = [impvol(S, K[i], T, r, Pr[i], 'C', 1e-8, 1e3) for i in np.arange(d)]

# moneyness
x = np.log(S / K)
plt.subplot(2,1,1)
#plt.plot(x, Pr, x, np.max([S * (np.exp(-x) - 1),np.zeros_like(x)], axis = 0))
plt.plot(np.log(K), Pr, np.log(K), np.max([S - K, np.zeros_like(K)], axis = 0))
plt.grid(1)
plt.subplot(2,1,2)
plt.plot(np.log(K), V)
plt.grid(1)
plt.show()

num = 6
phi = np.linspace(0., -.5, num)
lw = np.linspace(.0, .8, num)

for p, l in zip(phi, lw):
    P['phi'] = p
    # Price option according to the model
    Pr = COS(S, K, T, r, distr, P)
    # compute implied volatilities
    V = [impvol(S, K[i], T, r, Pr[i], 'C', 1e-8, 1e3) for i in np.arange(d)]

    plt.plot(np.log(K), V, color = str(l), linewidth = 2.)

plt.legend(phi)
#plt.scatter(subset['moneyness'], subset['impl_volatility'])
plt.show()

T = np.linspace(10., 60., num) / 365
P['phi'] = 0.

for t, l in zip(T, lw):
    P['T'] = t
    # Price option according to the model
    Pr = COS(S, K, t, r, distr, P)
    # compute implied volatilities
    V = [impvol(S, K[i], t, r, Pr[i], 'C', 1e-8, 1e3) for i in np.arange(d)]

    plt.plot(np.log(K), V, color = str(l), linewidth = 2.)

plt.legend(T*365)
#plt.scatter(subset['moneyness'], subset['impl_volatility'])
plt.show()
