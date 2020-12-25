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
