"""Implement the two layers neural network example of the paper."""
import numpy as np

import env
import losses as los
import activations as act


def layer1(theta, x, sigma):
    """Implement the first layer (the hidden layer) of the paper.

    Args:
    -----
        theta : np.array of shape (m, d)
        x : np.array of shape (n, d-1)
        sigma : callable
            Activation function

    Returns:
    --------
        np.array of shape (m, n)

    """
    return sigma(np.inner(theta[:, :-1], x) + theta[:, -1, None])


def phi(w, theta, x, sigma):
    """Implement the phi function of the paper.

    Args:
    -----
        w : np.array of shape (m,)
        theta : np.array of shape (m, d)
        x : np.array of shape (n, d-1)
        sigma : callable
            Activation function

    Returns:
    --------
        np.array of shape (m, n)

    """
    return w[:, None]*layer1(theta, x, sigma)  # (m, n)


def phi_dw(w, theta, x, sigma):
    """Implement the derivative of the phi function wrt w.

    Args:
    -----
        w : np.array of shape (m,)
        theta : np.array of shape (m, d)
        x : np.array of shape (n, d-1)
        sigma : callable
            Activation function

    Returns:
    --------
        np.array of shape (m, n)

    """
    return layer1(theta, x, sigma)  # (m, n)


def phi_dtheta1(w, theta, x, sigma_d):
    """Implement the derivative of the phi function wrt the first d-1 thetas.

    Args:
    -----
        w : np.array of shape (m,)
        theta : np.array of shape (m, d)
        x : np.array of shape (n, d-1)
        sigma_d : callable
            Derivative of the activation function

    Returns:
    --------
        np.array of shape (m, d, n)

    """
    return w[:, None, None]*x[None, :, :]*layer1(theta, x, sigma_d)[:, None, :]


def phi_dtheta2(w, theta, x, sigma_d):
    """Implement the derivative of the phi function wrt the last theta.

    Args:
    -----
        w : np.array of shape (m,)
        theta : np.array of shape (m, d)
        x : np.array of shape (n, d-1)
        sigma_d : callable
            Derivative of the activation function

    Returns:
    --------
        np.array of shape (m, n)

    """
    return w[:, None]*layer1(theta, x, sigma_d)  # (m, n)
