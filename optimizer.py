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


def subgrad_f_m(env, w, theta, n=1000):
    """Evaluate a subgradient of the objective f_m.

    Args:
    -----
        env : Env named tuple
        w : np.array of shape (m,)
        theta : np.array of shape (m, d)
        n : int
            Discretization for the integral computation

    Returns:
    --------
        subgrad_w : np.array of shape (m,)
        subgrad_theta : np.array of shape (m, d)

    """
    x = np.linspace(env.x_min, env.x_max, n)
    grad_R = env.grad_R(w, theta, x)
    subgrad_V = env.subgrad_V(w, theta)
    return grad_R[0] + subgrad_V[0], grad_R[1] + subgrad_V[1]


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


def forward_backward(env, w0, theta0, max_iter, n=1000, print_every=None):
    """Implement the forward backward algorithm to minimize f.

    Args:
    -----
        env : Env named tuple
        w0 : np.array of shape (m,)
        theta0 : np.array of shape (m, d)
        max_iter : int
        n : int
            Discretization for the integral computation
        print_every : int

    Returns:
    --------
        w : np.array of shape (m,)
        theta : np.array of shape (m, d)

    """
    w, theta = np.copy(w0), np.copy(theta0)
    ws, thetas = [], []

    # Parameters of the algorithm
    nu = 8e3/env.lbd  # gaussian
    # nu = 5e2/env.lbd  # paper
    gamma = 1.99/nu
    delta = 2 - gamma*nu/2
    lbd = 0.99*delta

    for k in range(max_iter):
        w, theta = forward_backward_step(env, w, theta, gamma, lbd, n)
        ws.append(w)
        thetas.append(theta)

        # Check subgradient and objective value
        if print_every is not None and k % print_every == 0:
            subgrad_w, subgrad_theta = subgrad_f_m(env, w, theta, n)
            e = np.linalg.norm(subgrad_w) + np.linalg.norm(subgrad_theta)
            fm = f_m(env, w, theta, n)

            print(f'iter {k}: \t e={e:.2e} \t fm={fm:.2e}')

    return np.array(ws), np.array(thetas)


def SGD(env, w0, theta0, bs, n_iter, gamma0):
    w, theta = np.copy(w0), np.copy(theta0)
    m = w.shape[0]

    for k in range(n_iter):
        print(f'iter {k}')

        # Sample a batch
        mean, cov = np.zeros(env.d), np.eye(env.d)
        x = np.random.multivariate_normal(mean, cov, size=bs)

        # Forward pass
        y_hat = env.forward(w, theta, x)
        loss_d = env.loss_d1(y_hat, env.y(x))

        grad_w = env.phi_dw(w, theta, x)*loss_d[:, None]/m
        grad_theta = env.phi_dtheta(w, theta, x)*loss_d[:, None, None]/m

        # Approx the expectancy by Monte Carlo
        grad_w = grad_w.mean(axis=0)
        grad_theta = grad_theta.mean(axis=0)

        gamma = gamma0/np.power(k+1, .75)

        w -= gamma*grad_w
        theta -= gamma*grad_theta

    return w, theta


