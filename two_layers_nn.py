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
        np.array of shape (n, m)

    """
    return sigma(np.inner(x, theta[:, :-1]) + theta[None, -1, :])


def layer2(w, theta, x, sigma):
    """Implement the second layer of the paper.

    Args:
    -----
        w : np.array of shape (m,)
        theta : np.array of shape (m, d)
        x : np.array of shape (n, d-1)
        sigma : callable
            Activation function

    Returns:
    --------
        np.array of shape (n,)

    """
    return np.mean(phi(w, theta, x, sigma), axis=0)


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
        np.array of shape (n, m)

    """
    return w[None, :]*layer1(theta, x, sigma)  # (n, m)


def phi_dw(theta, x, sigma):
    """Implement the derivative of the phi function wrt w.

    Args:
    -----
        theta : np.array of shape (m, d)
        x : np.array of shape (n, d-1)
        sigma : callable
            Activation function

    Returns:
    --------
        np.array of shape (n, m)

    """
    return layer1(theta, x, sigma)  # (n, m)


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
        np.array of shape (n, m, d-1)

    """
    return w[None, :, None]*x[:, None, :]*layer1(theta, x, sigma_d)[:, :, None]


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
        np.array of shape (n, m)

    """
    return w[None, :]*layer1(theta, x, sigma_d)  # (n, m)


def phi_dtheta(w, theta, x, sigma_d):
    """Implement the derivative of the phi function wrt theta.

    Args:
    -----
        w : np.array of shape (m,)
        theta : np.array of shape (m, d)
        x : np.array of shape (n, d-1)
        sigma_d : callable
            Derivative of the activation function

    Returns:
    --------
        np.array of shape (n, m)

    """
    d1 = phi_dtheta1(w, theta, x, sigma_d)
    d2 = phi_dtheta2(w, theta, x, sigma_d)
    return np.concatenate((d1, d2), axis=0)


def y(w, theta, x, sigma):
    """Implement the second layer of the paper.

    Args:
    -----
        w : np.array of shape (m,)
        theta : np.array of shape (m, d)
        x : np.array of shape (n, d-1)
        sigma : callable
            Activation function

    Returns:
    --------
        np.array of shape (n,)

    """
    return layer2(w, theta, x, sigma)


def V(w, theta):
    """Implement the second layer of the paper.

    Args:
    -----
        w : np.array of shape (m,)
        theta : np.array of shape (m, d)

    Returns:
    --------
        np.array of shape (m,)

    """
    return np.abs(w)*np.linalg.norm(theta, ord=1, axis=1)


def paper_env(m0, sigma, loss_name):
    d = 3

    # Generate ground truth
    w_bar = np.random.normal(0, 1, size=m0)
    theta_bar = np.random.multivariate_normal(0, np.eye(d-1), size=m0)

    loss, loss_d1 = los.get_loss(loss_name)

    return env.NNEnv(
        d=d-2,
        w_bar=w_bar,
        theta_bar=theta_bar,
        y=lambda x: y(w_bar, theta_bar, x, sigma),
        V=V,
        phi=lambda w, theta, x: phi(w, theta, x, sigma),
        phi_dw=lambda w, theta, x: phi_dw(theta, x, sigma),
        phi_dtheta=lambda w, theta, x: phi_dtheta(w, theta, x, sigma),
        loss=loss,
        loss_d1=loss_d1,
        forward=lambda w, theta, x: layer2(w, theta, x, sigma),
    )
