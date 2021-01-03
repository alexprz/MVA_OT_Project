"""Implement the sparse deconvolution example of the paper."""
import numpy as np

from env import Env
import kernels as ker


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


def draw_positions_1D(m):
    """Draw positions in (0, 1) with a minimum separaton of 0.1.

    Args:
    -----
        m : int
            Number of positions to draw.

    Returns:
    --------
        p : array of shape (m,)

    """
    odd = np.random.binomial(n=1, p=0.5)
    chunks = np.random.choice(np.arange(odd, 10, 2), size=m, replace=False)
    return (chunks + np.random.uniform(0, 1, size=m))/10


def y(x, w, theta, psi):
    """Implement the y function.

    Args:
    -----
        x : np.array of shape (N,) or scalar
        w : np.array of shape (m,)
        theta : np.array of shape (m,)
        psi : callable

    Returns:
    --------
        y : np.array of shape (N,)

    """
    m = w.shape[0]
    x = np.squeeze(np.array(x))
    _y = np.dot(psi(x[:, None] - theta[None, :]), w)
    # assert _y.shape == (x.shape[0], )
    return _y


def R(f, y, lbd):
    """Implement the loss function.

    Args:
    -----
        f : np.array of shape (d, N)
            N values of d functions f on (0, 1) equally spaced
        y : callable
        lbd : float

    Returns:
    --------
        float

    """
    f = np.array(f)
    f = f.reshape(1, -1) if len(f.shape) == 1 else f
    N = f.shape[1]
    linspace = np.linspace(0, 1, N)
    return 1/(2*lbd*N)*np.sum(np.power(f - y(linspace)[None, :], 2), axis=1)


def grad_R(w, theta, x, y, psi, psi_p, lbd):
    """Gradient of R wrt w and theta.

    Args:
    -----
        w : np.array of shape (m,)
        theta : np.array of shape (m,)
        x : np.array of shape (n,)
        y : callable
        psi : callable
        psi_p : callable
        lbd : float

    Returns:
    --------
        grad_w : np.array of shape (m,)
        grad_theta : np.array of shape (m,)

    """
    m = w.shape[0]
    theta = np.squeeze(theta)
    w = np.squeeze(w)
    x = np.squeeze(x)
    dx = x[:, None] - theta[None, :]  # (n, m)
    psi_t = psi(dx)  # (n, m)
    z = (y(x) - np.dot(psi_t, w)/m)  # (n,)
    grad_w = -np.mean(psi_t*z[:, None], axis=0)/(lbd*m)  # (m,)
    grad_theta = w*np.mean(psi_p(dx)*z[:, None], axis=0)/(lbd*m)  # (m,)

    return grad_w, grad_theta


def subgrad_V(w, theta):
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
    subgrad_w = np.sign(w)/m
    subgrad_theta = np.zeros_like(theta)

    return subgrad_w, subgrad_theta


def prox_V(w, theta, gamma):
    """Compute the proximity operator of gamma*V.

    Args:
    -----
        w : np.array of shape (m,)
        theta : np.array of shape (m,)
        gamma : float

    Returns:
        prox_w : np.array of shape (m,)
        prox_theta : np.array of shape (m,)

    """
    m = w.shape[0]
    prox_w = np.sign(w)*np.maximum(np.abs(w) - gamma/m, 0)
    prox_theta = theta

    return prox_w, prox_theta


def phi(w, theta, x, psi):
    """Implement the phi function.

    Args:
    -----
        w : np.array of shape (m,)
        theta : np.array of shape (m, d)
        x : np.array of shape (N, d)
        psi : callable

    Returns:
    --------
        np.array of shape (m, N)

    """
    x = x.reshape(-1, 1) if x.ndim == 1 else x
    theta = theta.reshape(-1, 1) if theta.ndim == 1 else theta
    w = np.squeeze(w)
    return w[:, None]*psi(x[None, :, :] - theta[:, None, :])


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


def paper_env(m0):
    """Create the same environment as in the paper for sparse deconvolution.

    Args:
    -----
        m0 : int
            Number of spikes

    Returns:
    --------
        Env namedtuple
    """
    signs = 2*np.random.binomial(n=1, p=0.5, size=m0) - 1  # weight signs
    w = signs*np.random.uniform(0.5, 1.5, size=m0)  # weights
    p = draw_positions_1D(m0)  # positions
    lbd = 0.2
    # lbd = 0.05

    def _g(x): return spikes_1D(x, w, p)  # ground truth
    def psi(x): return ker.dirichlet_kernel(2*np.pi*x, n=7)  # filter
    def psi_p(x): return ker.dirichlet_kernel_dx(2*np.pi*x, n=7)  # filter
    def _phi(w, theta, x): return phi(w, theta, x, psi)  # weighted translate
    def V(w, theta): return np.abs(w)  # regularization
    def _y(x): return y(x, w, p, psi)  # noisy observation
    def _R(f): return R(f, _y, lbd=lbd)
    def _grad_R(w, theta, x): return grad_R(w, theta, x, _y, psi, psi_p, lbd=lbd)

    x_min = np.array([0])
    x_max = np.array([1])

    return Env(R=_R, phi=_phi, V=V, y=_y, g=_g, w=w, p=p,
               x_min=x_min, x_max=x_max, grad_R=_grad_R, subgrad_V=subgrad_V,
               psi=psi, psi_p=psi_p, prox_V=prox_V, lbd=lbd)


def gaussian_env(m0, sigma):
    """Create a paper-like environment with a gaussian filter.

    Args:
    -----
        m0 : int
            Number of spikes
        sigma : float
            Variance of the gaussian kernel

    Returns:
    --------
        Env namedtuple
    """
    signs = 2*np.random.binomial(n=1, p=0.5, size=m0) - 1  # weight signs
    w = signs*np.random.uniform(0.5, 1.5, size=m0)  # weights
    p = draw_positions_1D(m0)  # positions
    lbd = 4
    # lbd = 0.05

    def _g(x): return spikes_1D(x, w, p)  # ground truth
    def psi(x): return ker.gaussian_kernel(x, sigma=sigma)  # filter
    def psi_p(x): return ker.gaussian_kernel_dx(x, sigma=sigma)  # filter
    def _phi(w, theta, x): return phi(w, theta, x, psi)  # weighted translate
    def V(w, theta): return np.abs(w)  # regularization
    def _y(x): return y(x, w, p, psi)  # noisy observation
    def _R(f): return R(f, _y, lbd=lbd)
    def _grad_R(w, theta, x): return grad_R(w, theta, x, _y, psi, psi_p, lbd=lbd)

    x_min = np.array([0])
    x_max = np.array([1])

    return Env(R=_R, phi=_phi, V=V, y=_y, g=_g, w=w, p=p,
               x_min=x_min, x_max=x_max, grad_R=_grad_R, subgrad_V=subgrad_V,
               psi=psi, psi_p=psi_p, prox_V=prox_V, lbd=lbd)
