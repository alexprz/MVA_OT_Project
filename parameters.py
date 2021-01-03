"""Implement classes that store parameters for the experiments."""
import numpy as np
from abc import ABC, abstractmethod

import kernels
import sparse_deconvolution_1D as sd1


class BaseParameters(ABC):
    """Abstract class for parameters classes."""

    def __str__(self):
        return


class SD1Parameters(BaseParameters):
    """Store parameters for the sparse deconvolution example."""

    def __init__(self, m0, w0, theta0, w_bar, theta_bar, lbd, kernel,
                 n_iter, fb_gamma, fb_lbd, fb_nu, n):
        """Init.

        Args:
        -----
            m0 : int
                Number of particles in the ground truth
            w0 : np.array of shape (m,)
                Initial weights
            theta0 : np.array of shape (m,)
                Initial positions
            w_bar : np.array of shape (m,)
                Groud truth weights
            theta_bar : np.array of shape (m,)
                Groud truth positions
            lbd : float
                Objective parameter
            kernel : kernels.BaseKernel object
            n_iter : int
                Number of iterations of the FB algorithm
            fb_gamma : float
                Parameter of the FB algo
            fb_lbd : float
                Parameter of the FB algo
            fb_nu : float
                Parameter of the FB algo
            n : int
                Discretization for the integral computation

        """
        self.m0 = m0
        self.m = w0.shape[0]
        self.w0 = w0
        self.theta0 = theta0
        self.w_bar = w_bar
        self.theta_bar = theta_bar
        self.lbd = lbd
        self.n_iter = n_iter
        self.kernel = kernel
        self.fb_gamma = fb_gamma
        self.fb_lbd = fb_lbd
        self.fb_nu = fb_nu
        self.n = n

        assert isinstance(kernel, kernels.BaseKernel)


class SD1CommonParameters(BaseParameters):
    """Store common parameters for the sparse deconvolution example."""

    def __init__(self, m, kernel):
        """Init.

        Args:
        -----
            kernel : kernels.BaseKernel object

        """
        np.random.seed(0)
        m0 = 5
        lbd = 0.2
        w_bar, theta_bar = sd1.paper_ground_truth(m0)
        super().__init__(
            m0=m0,
            w0=np.zeros(m),
            theta0=np.arange(m)/m,
            w_bar=w_bar,
            theta_bar=theta_bar,
            lbd=lbd,
            n_iter=10000,
            kernel=kernel,
            fb_gamma=1,
            fb_lbd=0.01,
            fb_nu=1/lbd,
            n=100,
        )


class XP11Params(SD1CommonParameters):
    """Implement the parameters for the experiment 1.1 (Dirichlet kernel)."""

    def __init__(self, m, order):
        """Init.

        Args:
        -----
            m : int
                Number of particles in the gradient flow
            order : int
                Order of the Dirichlet kernel

        """
        super().__init__(
            m=m,
            kernel=kernels.DirichletKernel(period=1, n=order),
        )


class XP12Params(SD1Parameters):
    """Implement the parameters for the experiment 1.2 (Gaussian kernel)."""

    def __init__(self, m, sigma):
        """Init.

        Args:
        -----
            m : int
                Number of particles in the gradient flow
            sigma : float
                Width of the Gaussian kernel

        """
        super().__init__(
            m=m,
            kernel=kernels.GaussianKernel(sigma),
        )
