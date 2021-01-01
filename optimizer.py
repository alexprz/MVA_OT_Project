"""Implement the optimizers that minimizes F."""
import numpy as np

import sparse_deconvolution_1D as sd1


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
    # nu = 8e3/env.lbd  # gaussian
    nu = 1/env.lbd
    # nu = 5e2/env.lbd  # paper
    gamma = 1/nu
    delta = 2 - gamma*nu/2
    lbd = 0.01*delta

    for k in range(max_iter):
        w, theta = forward_backward_step(env, w, theta, gamma, lbd, n)
        ws.append(w)
        thetas.append(theta)

        # Check subgradient and objective value
        if print_every is not None and k % print_every == 0:
            subgrad_w, subgrad_theta = sd1.subgrad_f_m(env, w, theta, n)
            e = np.linalg.norm(subgrad_w) + np.linalg.norm(subgrad_theta)
            fm = sd1.f_m(env, w, theta, n)

            print(f'iter {k}: \t e={e:.2e} \t fm={fm:.2e}')

    return np.array(ws), np.array(thetas)


def SGD(env, w0, theta0, bs, n_iter, gamma0, print_every=None):
    """Implement stochastic gradient descent for exemple 2."""
    w, theta = np.copy(w0), np.copy(theta0)
    ws, thetas = [np.copy(w)], [np.copy(theta)]

    fms, norm_grad_fms = [], []
    Rms, Vms = [], []
    m = w.shape[0]

    for k in range(n_iter):
        # Adjust step size
        gamma = gamma0/np.power(k+1, .75)

        # Sample a batch
        mean, cov = np.zeros(env.d), np.eye(env.d)
        x = np.random.multivariate_normal(mean, cov, size=bs)

        # Compute the gradient over the batch
        grad_w, grad_theta = env.grad_fm(w, theta, x)

        # Update point
        w -= m*gamma*grad_w
        theta -= m*gamma*grad_theta

        ws.append(np.copy(w))
        thetas.append(np.copy(theta))

        # Check new objective
        fm = env.fm(w, theta, x)
        fms.append(fm)
        Rm = env.Rm(w, theta, x)
        Rms.append(Rm)
        Vm = env.Vm(w, theta)
        Vms.append(Vm)

        e_w = np.linalg.norm(grad_w)
        e_theta = np.linalg.norm(grad_theta)
        e = e_w + e_theta
        norm_grad_fms.append(e)

        if print_every is not None and (k == 0 or (k+1) % print_every == 0):
            print(f'iter {k+1}: \t ∇w={e_w:.2e} \t ∇θ={e_theta:.2e} \t fm={fm:.2e} \t ∇fm={e:.2e}')

    ws = np.array(ws)
    thetas = np.array(thetas)
    fms = np.array(fms)
    norm_grad_fms = np.array(norm_grad_fms)
    Rms = np.array(Rms)
    Vms = np.array(Vms)
    return ws, thetas, fms, norm_grad_fms, Rms, Vms


