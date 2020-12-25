"""Implement the sparse deconvolution example of the paper."""
import numpy as np


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
    return np.sin((N + .5)*x)/(2*np.pi*np.sin(x/2))


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
