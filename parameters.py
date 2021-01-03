"""Implement classes that store parameters for the experiments."""
import numpy as np
from abc import ABC, abstractmethod

from kernels import BaseKernel, DirichletKernel, GaussianKernel
import sparse_deconvolution_1D as sd1


class BaseParameters(ABC):
    """Abstract class for parameters classes."""

    def __str__(self):
        """Give a str representation of the parameters."""
        return ''.join([
            f'-{k}_{v}' for k, v in self.state_dict().items() if v is not None
        ])

    @abstractmethod
    def state_dict(self):
        """Create a dict storing parameters."""
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

        self.name = None
        self.subname = None

        assert isinstance(kernel, BaseKernel)

    def state_dict(self):
        """Create a dict storing parameters."""
        return {
            'm0': self.m0,
            'm': self.m,
            'lbd': self.lbd,
            'n_iter': self.n_iter,
            'kernel': self.kernel,
            'fb_gamma': self.fb_gamma,
            'fb_lbd': self.fb_lbd,
            'fb_nu': self.fb_nu,
            'n': self.n,
            'which': self.subname,
        }


class SD1CommonParameters(SD1Parameters):
    """Store common parameters for the sparse deconvolution example."""

    def __init__(self, m, **kwargs):
        """Init.

        Args:
        -----
            kernel : kernels.BaseKernel object

        """
        np.random.seed(0)
        m0 = 5
        lbd = kwargs.get('lbd', 0.2)
        w_bar, theta_bar = sd1.paper_ground_truth(m0)
        super().__init__(
            m0=m0,
            w0=kwargs.get('w0', np.zeros(m)),
            theta0=kwargs.get('theta0', np.arange(m)/m),
            w_bar=w_bar,
            theta_bar=theta_bar,
            lbd=lbd,
            n_iter=10000,
            kernel=kwargs.get('kernel', DirichletKernel(period=1, n=7)),
            fb_gamma=kwargs.get('fb_gamma', 1),
            fb_lbd=kwargs.get('fb_lbd', 0.01),
            fb_nu=kwargs.get('fb_nu', 1/lbd),
            n=100,
        )


class XP11Params(SD1CommonParameters):
    """Implement the parameters for the experiment 1.1 (Dirichlet kernel)."""

    def __init__(self, m, order, **kwargs):
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
            kernel=DirichletKernel(period=1, n=order),
            **kwargs
        )
        self.name = 'XP1-1'


class XP12Params(SD1CommonParameters):
    """Implement the parameters for the experiment 1.2 (Gaussian kernel)."""

    def __init__(self, m, sigma, **kwargs):
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
            kernel=GaussianKernel(sigma),
            **kwargs
        )
        self.name = 'XP1-2'


class XP13Params(SD1CommonParameters):
    """Implement the parameters for the experiment 1.3 (lbd influence)."""

    def __init__(self, m, lbd, **kwargs):
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
            lbd=lbd,
            **kwargs
        )
        self.name = 'XP1-3'


class XP14Params(SD1CommonParameters):
    """Implement the parameters for the experiment 1.4 (init influence)."""

    def __init__(self, w0, theta0, name, **kwargs):
        """Init.

        Args:
        -----
            m : int
                Number of particles in the gradient flow
            sigma : float
                Width of the Gaussian kernel

        """
        assert w0.shape[0] == theta0.shape[0]
        super().__init__(
            m=w0.shape[0],
            w0=w0,
            theta0=theta0,
            **kwargs
        )
        self.name = 'XP1-4'
        self.subname = name
