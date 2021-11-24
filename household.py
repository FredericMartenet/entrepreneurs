"""Routines for the household side"""

import numpy as np
from scipy.stats import norm


def agrid(amax, n, amin=0):
    """Create grid between amin-pivot and amax+pivot that is equidistant in logs."""
    pivot = np.abs(amin) + 0.25
    a_grid = np.geomspace(amin + pivot, amax + pivot, n) - pivot
    a_grid[0] = amin  # make sure *exactly* equal to amin
    return a_grid


def pareto_cdf(x, eta=4.15):
    """"Pareto distribution CDF"""
    return 1 - x ** (-eta)


def pareto_cdfinv(x, eta=4.15):
    """"Pareto distribution inverse CDF"""
    return (1 - x) ** (- 1 / eta)


def pareto_discrete(eta, n, z_from=.36, z_to=.998, zlast=np.array([0.9985, 0.9990, 0.9992, .9995])):
    """Discretized Pareto distribution with custom end points."""
    if zlast is None:
        z = np.geomspace(start=pareto_cdfinv(z_from, eta=eta), stop=pareto_cdfinv(z_to, eta=eta), num=n)
    else:
        z = np.hstack((
            np.linspace(start=pareto_cdfinv(z_from, eta=eta), stop=pareto_cdfinv(z_to, eta=eta), num=n - len(zlast)),
            pareto_cdfinv(zlast, eta=eta)))
    pi_z = np.hstack((pareto_cdf(z[0], eta=eta) / pareto_cdf(z[-1], eta=eta),
                      (pareto_cdf(z[1:], eta=eta) - pareto_cdf(z[:-1], eta=eta)) / pareto_cdf(z[-1], eta=eta)))
    pi_z /= np.sum(pi_z)

    return z, pi_z


def stationary(Pi, pi_seed=None, tol=1E-11, maxit=10_000):
    """Find invariant distribution of a Markov chain by iteration."""
    if pi_seed is None:
        pi = np.ones(Pi.shape[0]) / Pi.shape[0]
    else:
        pi = pi_seed

    for it in range(maxit):
        pi_new = pi @ Pi
        if np.max(np.abs(pi_new - pi)) < tol:
            break
        pi = pi_new
    else:
        raise ValueError(f'No convergence after {maxit} forward iterations!')
    pi = pi_new

    return pi


def variance(x, pi):
    """Variance of discretized random variable with support x and probability mass function pi."""
    return np.sum(pi * (x - np.sum(pi * x)) ** 2)


def markov_tauchen(rho, sigma, N=7, m=3):
    """Tauchen method discretizing AR(1) s_t = rho*s_(t-1) + eps_t.

    Parameters
    ----------
    rho   : scalar, persistence
    sigma : scalar, unconditional sd of s_t
    N     : int, number of states in discretized Markov process
    m     : scalar, discretized s goes from approx -m*sigma to m*sigma

    Returns
    ----------
    y  : array (N), states proportional to exp(s) s.t. E[y] = 1
    pi : array (N), stationary distribution of discretized process
    Pi : array (N*N), Markov matrix for discretized process
    """
    # make normalized grid, start with cross-sectional sd of 1
    s = np.linspace(-m, m, N)
    ds = s[1] - s[0]
    sd_innov = np.sqrt(1 - rho ** 2)

    # standard Tauchen method to generate Pi given N and m
    Pi = np.empty((N, N))
    Pi[:, 0] = norm.cdf(s[0] - rho * s + ds / 2, scale=sd_innov)
    Pi[:, -1] = 1 - norm.cdf(s[-1] - rho * s - ds / 2, scale=sd_innov)
    for j in range(1, N - 1):
        Pi[:, j] = (norm.cdf(s[j] - rho * s + ds / 2, scale=sd_innov) -
                    norm.cdf(s[j] - rho * s - ds / 2, scale=sd_innov))

    # invariant distribution and scaling
    pi = stationary(Pi)
    s *= (sigma / np.sqrt(variance(s, pi)))
    y = np.exp(s) / np.sum(pi * np.exp(s))

    return y, pi, Pi


def markov_incomes(rho, sigma_y, N=11):
    """simple helper method that assumes AR(1) process in logs for incomes and
    scales aggregate income to 1, also that takes in sdy as the
    *cross-sectional* sd of log incomes"""
    sigma = sigma_y * np.sqrt(1 - rho ** 2)
    s, pi, Pi = markov_tauchen(rho, sigma, N)
    y = np.exp(s) / np.sum(pi * np.exp(s))

    return (y, pi, Pi)


def get_coh(w, z_grid, theta, pi, r, a_grid, Tr):
    """Returns matrix of cash-on-hand max(w, pi) + R*a + T"""
    return np.maximum(w * z_grid[:, np.newaxis] ** theta, pi) + (1 + r) * a_grid + Tr
