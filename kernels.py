"""Implement some kernels for the sparse deconvolution experiment."""
import numpy as np


def dirichlet_kernel(x, n):
    """Dirichlet kernel of given order (2 pi periodic).

    Args:
    -----
        x : float
        n : int
            Kernel order.

    Returns:
    --------
        float

    """
    x = np.squeeze(np.array(x))
    y = np.zeros(x.shape)
    indeterminate = np.mod(x, 2*np.pi) == 0
    y[indeterminate] = (2*n + 1)/(2*np.pi)
    y[~indeterminate] = np.divide(
        np.sin((n + .5)*x[~indeterminate]),
        (2*np.pi*np.sin(x[~indeterminate]/2))
    )
    return y


def dirichlet_kernel_dx(x, n):
    x = np.squeeze(np.array(x))
    y = np.zeros(x.shape)
    indeterminate = np.mod(x, 2*np.pi) == 0
    y[indeterminate] = 0

    z = x[~indeterminate]
    a = n + .5
    b = .5
    num = a*np.cos(a*z)*np.sin(b*z) - b*np.cos(b*z)*np.sin(a*z)
    denom = np.power(np.sin(b*z), 2)
    y[~indeterminate] = np.divide(num, denom)
    return y


def gaussian_kernel(x, sigma):
    """Gaussian kernel.

    Args:
        x : float or np.array
        sigma : float

    Returns:
    --------
        float or np.array

    """
    x = np.squeeze(x)
    return np.exp(-.5*np.power(x, 2)/sigma**2)/(sigma*np.sqrt(2*np.pi))


def gaussian_kernel_dx(x, sigma):
    """Derivative of Gaussian kernel.

    Args:
        x : float or np.array
        sigma : float

    Returns:
    --------
        float or np.array

    """
    x = np.squeeze(x)
    return -2*x*gaussian_kernel(x, sigma)/sigma**2
