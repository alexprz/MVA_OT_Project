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
    return sigma(np.inner(x, theta[:, :-1]) + theta[None, :, -1])


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
    return np.mean(phi(w, theta, x, sigma), axis=1)


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
    d2 = phi_dtheta2(w, theta, x, sigma_d)[:, :, None]
    return np.concatenate((d1, d2), axis=2)


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


# def V(w, theta):
#     """Implement the second layer of the paper.

#     Args:
#     -----
#         w : np.array of shape (m,)
#         theta : np.array of shape (m, d)

#     Returns:
#     --------
#         np.array of shape (m,)

#     """
#     return np.abs(w)*np.linalg.norm(theta, ord=1, axis=1)


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
    return np.abs(w)


def Vm(w, theta):
    """Implement the second layer of the paper.

    Args:
    -----
        w : np.array of shape (m,)
        theta : np.array of shape (m, d)

    Returns:
    --------
        np.array of shape (m,)

    """
    return V(w, theta).mean()


def V_dw(w, theta):
    """Implement the second layer of the paper.

    Args:
    -----
        w : np.array of shape (m,)
        theta : np.array of shape (m, d)

    Returns:
    --------
        np.array of shape (m,)

    """
    return np.sign(w)


def V_dtheta(w, theta):
    """Implement the second layer of the paper.

    Args:
    -----
        w : np.array of shape (m,)
        theta : np.array of shape (m, d)

    Returns:
    --------
        np.array of shape (m,)

    """
    return np.zeros_like(theta)


def f_m(env, w, theta, x):
    """Implement the discretized objective function F.

    Args:
    -----
        env : NNEnv named tuple
        w : np.array of shape (m,)
        theta : np.array of shape (m, d)
        x : np.array of shape (n, d-1)

    Returns:
    --------
        float

    """
    fm = env.Rm(w, theta, x) + env.Vm(w, theta)
    return fm.item()


def Rm(w, theta, x, y, loss, sigma):
    """Gradient of R wrt w and theta.

    Args:
    -----
        w : np.array of shape (m,)
        theta : np.array of shape (m, d)
        x : np.array of shape (n, d-1)
        y : callable
        sigma : callable
        sigma_d : callable

    Returns:
    --------
        grad_w : np.array of shape (n, m)
        grad_theta : np.array of shape (n, m)

    """
    y_hat = layer2(w, theta, x, sigma)
    return loss(y_hat, y(w, theta, x, sigma)).mean()


def grad_Rm(w, theta, x, y, phi_dw, phi_dtheta, loss_d1, sigma, sigma_d):
    """Gradient of R wrt w and theta.

    Args:
    -----
        w : np.array of shape (m,)
        theta : np.array of shape (m, d)
        x : np.array of shape (n, d-1)
        y : callable
        sigma : callable
        sigma_d : callable

    Returns:
    --------
        grad_w : np.array of shape (n, m)
        grad_theta : np.array of shape (n, m)

    """
    m = w.shape[0]
    y_hat = layer2(w, theta, x, sigma)
    loss_d = loss_d1(y_hat, y(w, theta, x, sigma))
    grad_w = phi_dw(theta, x, sigma)*loss_d[:, None]/m
    grad_theta = phi_dtheta(w, theta, x, sigma_d)*loss_d[:, None, None]/m

    return grad_w, grad_theta


def subgrad_Vm(w, theta):
    """Return a subgradient of V.

    Args:
    -----
        w : np.array of shape (m,)
        theta : np.array of shape (m,)

    Returns:
    --------
        subgrad_w : np.array of shape (m,)
        subgrad_theta : np.array of shape (m,)

    """
    m = w.shape[0]
    subgrad_w = V_dw(w, theta)
    subgrad_theta = V_dtheta(w, theta)

    return subgrad_w/m, subgrad_theta/m


def paper_env(m0, sigma_name, loss_name):
    d = 3

    # Generate ground truth
    w_bar = np.random.normal(0, 1, size=m0)
    mean, cov = np.zeros(d-1), np.eye(d-1)
    theta_bar = np.random.multivariate_normal(mean, cov, size=m0)

    loss, loss_d1 = los.get_loss(loss_name)
    sigma, sigma_d = act.get_sigma(sigma_name)

    return env.NNEnv(
        d=d-2,
        w_bar=w_bar,
        theta_bar=theta_bar,
        y=lambda x: y(w_bar, theta_bar, x, sigma),
        V=V,
        V_dw=V_dw,
        V_dtheta=V_dtheta,
        phi=lambda w, theta, x: phi(w, theta, x, sigma),
        phi_dw=lambda w, theta, x: phi_dw(theta, x, sigma),
        phi_dtheta=lambda w, theta, x: phi_dtheta(w, theta, x, sigma_d),
        loss=loss,
        loss_d1=loss_d1,
        forward=lambda w, theta, x: layer2(w, theta, x, sigma),
        sigma=sigma,
        sigma_d=sigma_d,
        Rm=lambda w, theta, x: Rm(w, theta, x, y, loss, sigma),
        Vm=Vm,
        grad_Rm=lambda w, theta, x: grad_Rm(w, theta, x, y, phi_dw, phi_dtheta, loss_d1, sigma, sigma_d),
        subgrad_Vm=subgrad_Vm,
    )
