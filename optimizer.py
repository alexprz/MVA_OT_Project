"""Implement the optimizers that minimizes F."""
import numpy as np


def f_m(env, w, theta, n=1000):
    """Implement the discretized objective function F.

    Args:
    -----
        env : Env named tuple
        w : np.array of shape (m,)
        theta : np.array of shape (m, d)
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

    fm = env.R(env.phi(w, theta, x).mean(axis=0)) + env.V(w, theta).mean()
    return fm.item()


def forward_backward_step(env, w, theta, gamma, lbd, n=1000):
    """Implement one step of the forward backward algo.

    Args:
    -----
        env : Env named tuple
        w : np.array of shape (m,)
        theta : np.array of shape (m, d)
        gamma : float
        lbd : float
        n : int
            Discretization for the integral computation

    Returns:
    --------
        w : np.array of shape (m,)
        theta : np.array of shape (m, d)

    """
    x = np.linspace(env.x_min, env.x_max, n)
    grad_w, grad_theta = env.grad_R(w, theta, x)

    w_aux = w - gamma*grad_w
    theta_aux = theta - gamma*grad_theta

    prox_w_aux, prox_theta_aux = env.prox_V(w_aux, theta_aux, gamma)

    w = w + lbd*(prox_w_aux - w)
    theta = theta + lbd*(prox_theta_aux - theta)

    return w, theta
