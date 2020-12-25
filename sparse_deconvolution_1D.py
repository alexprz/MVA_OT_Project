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
