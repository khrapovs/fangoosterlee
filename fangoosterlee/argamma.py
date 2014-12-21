#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 14:44:38 2012

@author: khrapovs
"""
from numpy import log, exp
import sys

def tail_rec(fun):
    def tail(fun):
        a = fun
        while callable(a):
            a = a()
        return a
    return (lambda n, v, u: tail(fun(n, v, u)))

class TailRecurseException:
  def __init__(self, args, kwargs):
    self.args = args
    self.kwargs = kwargs

def tail_call_optimized(g):
  """
  This function decorates a function with tail call
  optimization. It does this by throwing an exception
  if it is it's own grandparent, and catching such
  exceptions to fake the tail call optimization.
  
  This function fails if the decorated
  function recurses in a non-tail context.
  """
  def func(*args, **kwargs):
    f = sys._getframe()
    if f.f_back and f.f_back.f_back \
        and f.f_back.f_back.f_code == f.f_code:
      raise TailRecurseException(args, kwargs)
    else:
      while 1:
        try:
          return g(*args, **kwargs)
        except TailRecurseException, e:
          args = e.args
          kwargs = e.kwargs
  func.__doc__ = g.__doc__
  return func

#def psin_f(n, u, v, lfQ, gfQ):
#    if n == 0:
#        return v
#    else:
#        return ( lambda: psin_f(n-1, u, lfQ(v, u), lfQ, gfQ) )
#
#psin = tail_rec(psin_f)
#
#def upsn_f(n, u, v, lfQ, gfQ):
#    if n == 0:
#        return v
#    else:
#        return ( lambda: upsn_f(n-1, u, v + gfQ(psin(n-1, u, v, lfQ, gfQ), u), lfQ, gfQ) )
#
#upsn = tail_rec(upsn_f)

def CFARG(u, P):
    T = P['T']
    r = P['r']
    rho = P['rho']
    delta = P['delta']
    dailymean = P['dailymean']
    scale = dailymean * (1 - rho) / delta
    betap = rho / scale
    v0 = P['v0']
    theta1 = P['theta1']
    theta2 = P['theta2']
    phi = P['phi']
    n = int(T * 365)

    # Prices of risk
#    theta2 = .5 * (3 * phi + 1) / (phi + 1)
#    tau0 = - phi + (1 + phi) * (theta2 - .5)
#    tau = rho * phi * (1 - (1 + phi) * (theta2 - .5))
#    dau = scale * delta * phi * (1 - (1 + phi) * (theta2 - .5))

    # distribution of one period volatility
    a = lambda u: rho * u / (1 + scale * u)
    b = lambda u: delta * log(1 + scale * u)
    
    # 1-period joint distribution of returns and volatility
#    alpha = lambda v: - (- v * (tau + phi) + .5 * v**2 * (1 + phi))
#    beta  = lambda v: rho * phi * (- v + .5 * v**2 * (1 + phi))
#    gamma = lambda v: scale * delta * phi * (- v + .5 * v**2 * (1 + phi))
    
#    alpha = lambda v: (v * (tau0 + phi) - .5 * v**2 * (1 + phi)) / 365
#    
#    beta = lambda v: ((tau - rho * phi) * v \
#        + .5 * rho * phi * (1 + phi) * v**2) / 365
#    
#    gamma = lambda v: ((dau - scale * delta * phi) * v \
#        + .5 * scale * delta * phi * (1 + phi) * v**2) / 365

    center = phi / (scale * (1 + rho))**.5
    
    alpha = lambda v: (((theta2 - .5) * (1 - phi**2) + center) * v \
        - .5 * v**2 * (1 - phi**2) )

    # Risk-neutral parameters
    factor = 1 / (1 + scale * (theta1 + alpha(theta2)))
    scale_star = scale * factor
    betap_star = betap * scale_star / scale
    rho_star = scale_star * betap_star
    
    a_star = lambda u: rho_star * u / (1 + scale_star * u)
    b_star = lambda u: delta * log(1 + scale_star * u)

    beta  = lambda v: v * a_star(- center)
    gamma = lambda v: v * b_star(- center)


#    alpha = lambda v: .5 * v * (1 + phi) * (2 * theta2 - 1 - v) / 365
#    beta = lambda v: - rho * phi * alpha(v)
#    gamma = lambda v: - scale * delta * phi * alpha(v)

    # joint distribution
    lf = lambda u, v: a(u + alpha(v)) + beta(v)
    gf = lambda u, v: b(u + alpha(v)) + gamma(v)
    
    # risk-neutral 1-period joint distribution
    lfQ = lambda u, v: lf(theta1 + u, theta2 + v) - lf(theta1, theta2)
    gfQ = lambda u, v: gf(theta1 + u, theta2 + v) - gf(theta1, theta2)
    # n-period cumulative return distribution
    psin = lambda n, v: lfQ(0, v) if n == 1 else lfQ(psin(n-1, v), v)
    upsn = lambda n, v: gfQ(0, v) if n == 1 else gfQ(psin(n-1, v), v) \
        + upsn(n-1, v)
    
#    def psin_f(n, u, v = 0.):
#        if n == 0:
#            return v
#        else:
#            return ( lambda: psin_f(n-1, u, lfQ(v, u)) )
#    
#    psin = tail_rec(psin_f)
#    
#    def upsn_f(n, u, v = 0.):
#        if n == 0:
#            return v
#        else:
#            return ( lambda: upsn_f(n-1, u, v + gfQ(psin(n-1, u, v), u)) )
#    
#    upsn = tail_rec(upsn_f)
    
#    @tail_call_optimized
#    def psin(n, u, v = 0.):
#        if n == 0:
#            return v
#        else:
#            return psin(n-1, u, lfQ(v, u))
#    
#    @tail_call_optimized
#    def upsn(n, u, v = 0.):
#        if n == 0:
#            return v
#        else:
#            return upsn(n-1, u, v + gfQ(psin(n-1, u, v), u))
    
    # Unconditionl return cumulant generating function
    #Lambda = lambda n, v: -c(psin(n, -v)) - upsn(n, -v)
    #psi = exp(- psin(n, -1j * u, 0, lfQ, gfQ) * v0 - upsn(n, -1j * u, 0, lfQ, gfQ))
    psi = exp( - psin(n, -1j * u) * v0 - upsn(n, -1j * u) )
    psi = psi * exp(- 1j * u * r * T)
    #psi = lambda n, u: exp(- psin(n, -1j * u) * v0 - upsn(n, -1j * u))
    
    return psi


#P = {'T': 730.0/365, 'r': 0, 'rho': .9, 'delta': 1.1, 'dailymean': .3**2, \
#            'v0': .3**2, 'theta1': 0, 'theta2': 0, 'phi': -.9}
#
#u = 10j
#
#print CFARG(u, P)

#f = CFARG(u, P)