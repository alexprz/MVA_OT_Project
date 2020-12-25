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
    """Dirichlet kernel of given order.

    Args:
    -----
        x : float
        n : int
            Kernel order.

    Returns:
    --------
        float

    """
    return np.divide(np.sin((n + .5)*x), (2*np.pi*np.sin(x/2)))


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
    chunks = np.random.choice(10, size=m0, replace=False)
    return (chunks + np.random.uniform(0, 1, size=m0))/10


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
        f : np.array of shape (N,)
            Values of f on (0, 1) equally spaced
        y : callable
        lbd : float

    Returns:
    --------
        float

    """
    N = f.shape[0]
    linspace = np.linspace(0, 1, N)
    return 1/(2*lbd*N)*np.sum(np.power(f - y(linspace), 2))


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
    w = np.random.uniform(0.5, 1.5, size=m0)  # weights
    p = draw_positions_1D(m0)  # positions

    def g(x): return spikes_1D(x, w, p)  # ground truth
    def psi(x): return dirichlet_kernel(x, n=7)  # filter
    def _phi(w, theta): return phi(w, theta, psi)  # weighted translate
    def V(w, theta): return np.abs(w)  # regularization
    def _y(x): return y(x, w, p, psi)  # noisy observation
    def _R(f): return R(f, y, lbd=1)

    return Env(R=_R, phi=_phi, V=V, y=_y, g=_g)
