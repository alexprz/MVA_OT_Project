"""Implement the optimizers that minimizes F."""
import numpy as np
from numpy.linalg import norm


def forward_backward_step(env, w, theta):
    """Implement one step of the forward backward algo.

    Args:
    -----
        env : sd1.SparseDeconvolution object
        w : np.array of shape (m,)
        theta : np.array of shape (m, d)

    Returns:
    --------
        w : np.array of shape (m,)
        theta : np.array of shape (m, d)

    """
    # Retrieve algorithm's parameters
    gamma = env.params.fb_gamma/env.params.fb_nu
    delta = 2 - gamma*env.params.fb_nu/2
    lbd = env.params.fb_lbd*delta

    x = env.discretize()
    grad_w, grad_theta = env.grad_R(w, theta, x)

    w_aux = w - gamma*grad_w
    theta_aux = theta - gamma*grad_theta

    prox_w_aux, prox_theta_aux = env.prox_V(w_aux, theta_aux, gamma)

    w = w + lbd*(prox_w_aux - w)
    theta = theta + lbd*(prox_theta_aux - theta)

    return w, theta


def forward_backward(env, print_every=None):
    """Implement the forward backward algorithm to minimize f.

    Args:
    -----
        env : sd1.SparseDeconvolution object
        print_every : int

    Returns:
    --------
        w : np.array of shape (m,)
        theta : np.array of shape (m, d)

    """
    w, theta = np.copy(env.params.w0), np.copy(env.params.theta0)
    ws, thetas = [], []

    # Parameters of the algorithm
    # nu = 8e3/env.lbd  # gaussian
    # nu = 1/env.params.lbd
    # nu = 5e2/env.lbd  # paper
    # gamma = 1/nu
    # delta = 2 - gamma*nu/2
    # lbd = 0.01*delta

    for k in range(env.params.n_iter):
        w, theta = forward_backward_step(env, w, theta)
        ws.append(w)
        thetas.append(theta)

        subgrad_w, subgrad_theta = env.subgrad_f_m(w, theta)  # , env.params.n)
        e = np.linalg.norm(subgrad_w) + np.linalg.norm(subgrad_theta)

        # Check subgradient and objective value
        if print_every is not None and k % print_every == 0:
            fm = env.f_m(w, theta, env.params.n)
            print(f'iter {k}: \t |∇fm|=={e:.2e} \t fm={fm:.2e}')

        if env.params.tol is not None and e < env.params.tol:
            if env.params.n_min is not None and env.params.n_min > k:
                continue
            print(f'Converged to tolerance {env.params.tol:.2e} in {k} iter.')
            break

    return np.array(ws), np.array(thetas)


def SGD(env, print_every=None):
    """Implement stochastic gradient descent for exemple 2.

    Args:
    -----
        env : tln.TwoLayerNetwork object
        print_every : int

    Returns:
    --------
        w : np.array of shape (m,)
        theta : np.array of shape (m, d)

    """
    w, theta = np.copy(env.params.w0), np.copy(env.params.theta0)
    ws, thetas = [np.copy(w)], [np.copy(theta)]

    fms, norm_grad_fms = [], []
    Rms, Vms = [], []
    norm_grad_Rms, norm_grad_Vms = [], []
    m = w.shape[0]

    for k in range(env.params.sgd_n_iter):
        # Sample a batch
        x = np.random.normal(0, 1, size=(env.params.sgd_bs, env.d))
        x /= norm(x, axis=1)[:, None]

        # Adjust step size
        gamma = env.params.sgd_gamma   #/np.power(k+1, .51)

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
        grad_w_rm, grad_theta_rm = env.grad_Rm(w, theta, x)
        grad_w_vm, grad_theta_vm = env.subgrad_Vm(w, theta)
        norm_grad_Rms.append(norm(grad_w_rm) + norm(grad_theta_rm))
        norm_grad_Vms.append(norm(grad_w_vm) + norm(grad_theta_vm))

        e_w = norm(grad_w)
        e_theta = norm(grad_theta)
        e = e_w + e_theta
        norm_grad_fms.append(e)

        if print_every is not None and (k == 0 or (k+1) % print_every == 0):
            print(f'iter {k+1}: \t |∇w|={e_w:.2e} \t |∇θ|={e_theta:.2e} \t fm={fm:.3e} \t |∇fm|={e:.2e}')

    ws = np.array(ws)
    thetas = np.array(thetas)
    fms = np.array(fms)
    norm_grad_fms = np.array(norm_grad_fms)
    Rms = np.array(Rms)
    Vms = np.array(Vms)
    norm_grad_Rms = np.array(norm_grad_Rms)
    norm_grad_Vms = np.array(norm_grad_Vms)
    return ws, thetas, fms, norm_grad_fms, Rms, Vms, norm_grad_Rms, norm_grad_Vms
