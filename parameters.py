"""Implement classes that store parameters for the experiments."""
import numpy as np
from abc import ABC, abstractmethod

import kernels


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


class SD1PaperParams(SD1Parameters):
    """Implement the parameters of the paper for sparse deconvolution."""

    def __init__(self, m, w_bar, theta_bar, lbd, fb_gamma, fb_lbd, fb_nu, n, order=7):
        """Init.

        Args:
        -----
            m : int
                Number of particles in the gradient flow
            lbd : float
                Objective parameter
            fb_gamma : float
                Parameter of the FB algo
            fb_lbd : float
                Parameter of the FB algo
            fb_nu : float
                Parameter of the FB algo
            n : int
                Discretization for the integral computation
            order : int
                Order of the Dirichlet kernel

        """
        super().__init__(
            m0=5,
            w0=np.zeros(m),
            theta0=np.arange(m)/m,
            w_bar=w_bar,
            theta_bar=theta_bar,
            lbd=lbd,
            kernel=kernels.DirichletKernel(period=1, n=order),
            n_iter=10000,
            fb_gamma=fb_gamma,
            fb_lbd=fb_lbd,
            fb_nu=fb_nu,
            n=n
        )


class SD1GaussianParams(SD1Parameters):
    """Implement custom parameters with gaussian for sparse deconvolution."""

    def __init__(self, m, w_bar, theta_bar, lbd, fb_gamma, fb_lbd, fb_nu, sigma, n):
        """Init.

        Args:
        -----
            m : int
                Number of particles in the gradient flow
            lbd : float
                Objective parameter
            fb_gamma : float
                Parameter of the FB algo
            fb_lbd : float
                Parameter of the FB algo
            fb_nu : float
                Parameter of the FB algo
            sigma : float
                Width of the Gaussian kernel
            n : int
                Discretization for the integral computation

        """
        super().__init__(
            m0=5,
            w0=np.zeros(m),
            theta0=np.arange(m)/m,
            w_bar=w_bar,
            theta_bar=theta_bar,
            lbd=lbd,
            kernel=kernels.GaussianKernel(sigma),
            n_iter=10000,
            fb_gamma=fb_gamma,
            fb_lbd=fb_lbd,
            fb_nu=fb_nu,
            n=n
        )
