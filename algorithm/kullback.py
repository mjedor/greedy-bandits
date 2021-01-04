from math import log, sqrt, exp
import numpy as np

eps = 1e-15


def kl_bern(x, y):
    """ Kullback-Leibler divergence for Bernoulli distributions."""
    x = min(max(x, eps), 1-eps)
    y = min(max(y, eps), 1-eps)
    return x*log(x/y) + (1-x)*log((1-x)/(1-y))


def kl_poisson(x, y):
    """ Kullback-Leibler divergence for Poison distributions."""
    x = max(x, eps)
    y = max(y, eps)
    return y-x+x*log(x/y)


def kl_gamma(x, y, a=1):
    """ Kullback-Leibler divergence for gamma distributions."""
    x = max(x, eps)
    y = max(y, eps)
    return a*(x/y - 1 - log(x/y))


def kl_neg_bin(x, y, r=1):
    """ Kullback-Leibler divergence for negative binomial distributions."""
    return r*log((r+x)/(r+y)) - x * log(y*(r+x)/(x*(r+y)))


def kl_gauss(x, y, sig2=1.):
    """ Kullback-Leibler divergence for Gaussian distributions."""
    return (x - y) ** 2 / (2 * sig2)


def klucb(x, d, div, upperbound, lowerbound=-float('inf'), precision=1e-6):
    """The generic klUCB index computation.

    Input args.: x, d, div, upperbound, lowerbound=-float('inf'), precision=1e-6,
    where div is the KL divergence to be used.
    """
    low = max(x, lowerbound)
    up = upperbound
    while up-low > precision:
        m = (low+up)/2
        if div(x, m) > d:
            up = m
        else:
            low = m
    return (low+up)/2


def klucb_gauss(x, d, sig2=1., precision=0.):
    """klUCB index computation for Gaussian distributions.

    Note that it does not require any search.
    """
    return x + sqrt(2*sig2*d)


def klucb_poisson(x, d, precision=1e-6):
    """klUCB index computation for Poisson distributions."""
    upperbound = x+d+sqrt(d*d+2*x*d)  # looks safe, to check: left (Gaussian) tail of Poisson dev
    return klucb(x, d, kl_poisson, upperbound, precision)


def klucb_bern(x, d, precision=1e-6):
    """klUCB index computation for Bernoulli distributions."""
    upperbound = min(1., klucb_gauss(x, d))
    return klucb(x, d, kl_bern, upperbound, precision)


def klucb_exp(x, d, precision=1e-6):
    """klUCB index computation for exponential distributions."""
    if d < 0.77:
        upperbound = x/(1+2./3*d-sqrt(4./9*d*d+2*d))  # safe, klexp(x,y) >= e^2/(2*(1-2e/3)) if x=y(1-e)
    else:
        upperbound = x*exp(d+1)
    if d > 1.61:
        lowerbound = x*exp(d)
    else:
        lowerbound = x/(1+d-sqrt(d*d+2*d))
    return klucb(x, d, kl_gamma, upperbound, lowerbound, precision)
