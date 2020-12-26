"""Implement the optimizers that minimizes F."""
import numpy as np


def f_m(w, theta, env, n=1000):
    """Implement the discretized objective function F.

    Args:
    -----
        w : np.array of shape (m,)
        theta : np.array of shape (m, d)
        env : Env named tuple
        n : int
            Discretization for the integral computation

    Returns:
    --------
        float

    """
    if env.x_min.ndim == 1:
        x = np.linspace(0, 1, n)
    else:
        raise NotImplementedError('Add case ndim > 1')

    return env.R(env.phi(w, theta, x).mean(axis=0)) + env.V(w, theta).mean()
