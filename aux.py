import numpy as np
from numba import jit
from numba import guvectorize


def stationary(Pi, pi=None, atol=1E-10, maxit=10000):
    """computes stationary distribution of Markov chain with transition Pi via
    simple iteration until subsequent iterations have difference below atol.
    option to supply starting iteration pi either to speed up or, when there
    are multiple stationary distributions, to select one"""
    if pi is None:
        pi = np.ones(Pi.shape[0]) / Pi.shape[0]

    for it in range(maxit):
        pi_new = pi @ Pi
        if it % 10 == 0 and np.max(np.abs(pi_new - pi)) < atol:
            break
        pi = pi_new

    return pi_new


def invdist(a, a_pol, Pi, D=None, atol=1E-10, maxit=10000):
    """finds invariant distribution given s*a policy array a_pol for endogenous
    state a and Markov transition matrix Pi for exogenous state s, possibly
    with starting distribution a*s D as a seed"""
    pi = stationary(Pi)  # compute separately exogenous inv dist to start there
    if D is None:
        D = pi[:, np.newaxis] * np.ones_like(a) / a.shape[0]  # assume equispaced on grid

    a_pol_i, a_pol_pi = interpolate_coord(a, a_pol)  # obtain policy rule

    # now iterate until convergence according to atol, only checking every 10 it
    for it in range(maxit):
        Dnew = forward_iterate(D, Pi, a_pol_i, a_pol_pi)
        if it % 10 == 0 and np.max(np.abs(Dnew - D)) < atol:
            break
        D = Dnew

    return Dnew


@guvectorize(['void(float64[:], float64[:], float64[:], float64[:])'], '(n),(nq),(n)->(nq)')
def interpolate_y(x, xq, y, yq):
    """Efficient linear interpolation exploiting monotonicity.

    Complexity O(n+nq), so most efficient when x and xq have comparable number of points.
    Extrapolates linearly when xq out of domain of x.

    Parameters
    ----------
    x  : array (n), ascending data points
    xq : array (nq), ascending query points
    y  : array (n), data points

    Returns
    ----------
    yq : array (nq), interpolated points
    """
    nxq, nx = xq.shape[0], x.shape[0]

    xi = 0
    x_low = x[0]
    x_high = x[1]
    for xqi_cur in range(nxq):
        xq_cur = xq[xqi_cur]
        while xi < nx - 2:
            if x_high >= xq_cur:
                break
            xi += 1
            x_low = x_high
            x_high = x[xi + 1]

        xqpi_cur = (x_high - xq_cur) / (x_high - x_low)
        yq[xqi_cur] = xqpi_cur * y[xi] + (1 - xqpi_cur) * y[xi + 1]


@guvectorize(['void(float64[:], float64[:], uint32[:], float64[:])'], '(n),(nq)->(nq),(nq)')
def interpolate_coord(x, xq, xqi, xqpi):
    """Get representation xqi, xqpi of xq interpolated against x:
    xq = xqpi * x[xqi] + (1-xqpi) * x[xqi+1]

    Parameters
    ----------
    x    : array (n), ascending data points
    xq   : array (nq), ascending query points

    Returns
    ----------
    xqi  : array (nq), indices of lower bracketing gridpoints
    xqpi : array (nq), weights on lower bracketing gridpoints
    """
    nxq, nx = xq.shape[0], x.shape[0]

    xi = 0
    x_low = x[0]
    x_high = x[1]
    for xqi_cur in range(nxq):
        xq_cur = xq[xqi_cur]
        while xi < nx - 2:
            if x_high >= xq_cur:
                break
            xi += 1
            x_low = x_high
            x_high = x[xi + 1]

        xqpi[xqi_cur] = (x_high - xq_cur) / (x_high - x_low)
        xqi[xqi_cur] = xi


def forward_iterate(D, Pi, a_pol_i, a_pol_pi):
    """iterates from distribution D of s*a to Dnew, given interpolated
    endogenous policy rule given by (a_pol_i,a_pol_pi), and exogenous Markov
    transition matrix Pi for the exogenous state"""
    # carry forward interpolated policy to adjust distribution across 'a'
    Dnew = np.empty_like(D)
    for s in range(D.shape[0]):
        Dnew[s, :] = forward_policy(D[s, :], a_pol_i[s, :], a_pol_pi[s, :])

    # now use transition matrix to adjust distribution across 's'
    Dnew = Pi.T @ Dnew

    return Dnew


@jit
def forward_policy(D, xdi, xdpi):
    n = D.shape[0]
    Dnew = np.zeros_like(D)
    for i in range(n):
        it = xdi[i]
        Dnew[it] += xdpi[i] * D[i]
        Dnew[it + 1] += (1 - xdpi[i]) * D[i]
    return Dnew
