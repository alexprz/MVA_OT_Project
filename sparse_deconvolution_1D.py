"""Implement the sparse deconvolution example of the paper."""
import numpy as np

from env import Env


def spikes_1D(x, w, p):
    """1D weighted spikes.

    Args:
    -----
        x : float
        w : array of shape (m,)
            Weights of the spikes.
        p : array of shape (m,)
            Positions of the spikes.

    Returns:
    --------
        float

    """
    assert w.shape == p.shape
    assert len(w.shape) == 1

    return np.sum(w[p == x])


def dirichlet_kernel(x, n):
    """Dirichlet kernel of given order (2 pi periodic).

    Args:
    -----
        x : float
        n : int
            Kernel order.

    Returns:
    --------
        float

    """
    x = np.array(x)
    y = np.zeros(x.shape)
    indeterminate = np.mod(x, 2*np.pi) == 0
    y[indeterminate] = 2*n + 1
    y[~indeterminate] = np.divide(
        np.sin((n + .5)*x[~indeterminate]),
        (2*np.pi*np.sin(x[~indeterminate]/2))
    )
    return y


def draw_positions_1D(m):
    """Draw positions in (0, 1) with a minimum separaton of 0.1.

    Args:
    -----
        m : int
            Number of positions to draw.

    Returns:
    --------
        p : array of shape (m,)

    """
    odd = np.random.binomial(n=1, p=0.5)
    chunks = np.random.choice(np.arange(odd, 11, 2), size=m, replace=False)
    return (chunks + np.random.uniform(0, 1, size=m))/10


def y(x, w, p, psi):
    """Implement the y function.

    Args:
    -----
        x : np.array of shape (N,) or scalar
        w : np.array of shape (m,)
        p : np.array of shape (m,)
        psi : callable

    Returns:
    --------
        y : np.array of shape (N,)

    """
    x = np.array(x)
    _y = np.dot(psi(x[:, None] - p[None, :]), w)
    assert _y.shape == (x.shape[0], )
    return _y


def R(f, y, lbd):
    """Implement the loss function.

    Args:
    -----
        f : np.array of shape (d, N)
            N values of d functions f on (0, 1) equally spaced
        y : callable
        lbd : float

    Returns:
    --------
        float

    """
    f = np.array(f)
    f = np.reshape(1, -1) if len(f.shape) == 1 else f
    N = f.shape[1]
    linspace = np.linspace(0, 1, N)
    return 1/(2*lbd*N)*np.sum(np.power(f - y(linspace)[None, :], 2), axis=1)


def phi(w, theta, psi):
    """Implement the phi function.

    Args:
    -----
        w : np.array of shape (m,)
        theta : np.array of shape (d,)

    Returns:
    --------
        callable np.array of shape (d,) -> np.array of shape (m,)

    """
    return lambda x: psi(x - theta)*w


def paper_env(m0):
    """Create the same environment as in the paper for sparse deconvolution.

    Args:
    -----
        m0 : int
            Number of spikes

    Returns:
    --------
        Env namedtuple
    """
    signs = 2*np.random.binomial(n=1, p=0.5, size=m0) - 1  # weight signs
    w = signs*np.random.uniform(0.5, 1.5, size=m0)  # weights
    p = draw_positions_1D(m0)  # positions

    def _g(x): return spikes_1D(x, w, p)  # ground truth
    def psi(x): return dirichlet_kernel(2*np.pi*x, n=7)  # filter
    def _phi(w, theta): return phi(w, theta, psi)  # weighted translate
    def V(w, theta): return np.abs(w)  # regularization
    def _y(x): return y(x, w, p, psi)  # noisy observation
    def _R(f): return R(f, _y, lbd=1)

    return Env(R=_R, phi=_phi, V=V, y=_y, g=_g, w=w, p=p)
