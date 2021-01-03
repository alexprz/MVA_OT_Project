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

    def __init__(self, m0, w0, theta0, lbd, kernel, n_iter, fb_gamma, fb_lbd):
        """Init.

        Args:
        -----
            m0 : int
                Number of particles in the ground truth
            w0 : np.array of shape (m,)
                Groud truth weights
            theta0 : np.array of shape (m,)
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

        """
        self.m0 = m0
        self.n_iter = n_iter
        self.w0 = w0
        self.theta0 = theta0
        self.lbd = lbd
        self.fb_gamma = fb_gamma
        self.fb_lbd = fb_lbd
        self.kernel = kernel

        assert isinstance(kernel, kernels.BaseKernel)


class SD1PaperParams(SD1Parameters):
    """Implement the parameters of the paper for sparse deconvolution."""

    def __init__(self, m, lbd, fb_gamma, fb_lbd, n=7):
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
            n : int
                Order of the Dirichlet kernel

        """
        self.m0 = 5
        self.n_iter = 10000
        self.w0 = np.zeros(m)
        self.theta0 = np.arange(m)/m
        self.lbd = lbd
        self.fb_gamma = fb_gamma
        self.fb_lbd = fb_lbd
        self.kernel = kernels.DirichletKernel(period=1, n=n)


class SD1GaussianParams(SD1Parameters):
    """Implement custom parameters with gaussian for sparse deconvolution."""

    def __init__(self, m, lbd, fb_gamma, fb_lbd, sigma):
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
            sigma : float
                Width of the Gaussian kernel

        """
        self.m0 = 5
        self.n_iter = 10000
        self.w0 = np.zeros(m)
        self.theta0 = np.arange(m)/m
        self.lbd = lbd
        self.fb_gamma = fb_gamma
        self.fb_lbd = fb_lbd
        self.kernel = kernels.GaussianKernel(sigma)
