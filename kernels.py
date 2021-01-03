"""Implement some kernels for the sparse deconvolution experiment."""
import numpy as np
from abc import ABC, abstractmethod


class BaseKernel(ABC):
    """Abstract class for kernels."""

    def __call__(self, x):
        """Shortcut to call the evaluate function."""
        return self.evaluate(x)

    def __str__(self):
        """Give a str representation of the parameters."""
        return ''.join([f'-{k}_{v}' for k, v in self.state_dict().items()])

    @staticmethod
    @abstractmethod
    def evaluate(x):
        """Implement the kernel."""
        return

    @staticmethod
    @abstractmethod
    def derivative(x):
        """Implement the derivative of the kernel."""
        return

    @abstractmethod
    def state_dict(self):
        """Create a dict storing parameters."""
        return


class DirichletKernel(BaseKernel):
    """Implement dirichlet kernel of given order (2 pi periodic)."""

    def __init__(self, period, n):
        """Init.

        Args:
        -----
            n : int
                Order of the Dirichlet kernel.
            period : float
                Period of the Dirichlet kernel.

        """
        assert isinstance(n, int)
        self.n = n
        self.period = period

    def state_dict(self):
        """Create a dict storing parameters."""
        return {
            'order': self.n,
            'period': self.period,
        }

    def _dirichlet(self, x):
        x = np.squeeze(np.array(x))
        y = np.zeros(x.shape)
        indeterminate = np.mod(x, 2*np.pi) == 0
        y[indeterminate] = (2*self.n + 1)/(2*np.pi)
        y[~indeterminate] = np.divide(
            np.sin((self.n + .5)*x[~indeterminate]),
            (2*np.pi*np.sin(x[~indeterminate]/2))
        )
        return y

    def evaluate(self, x):
        """Evaluate the kernel.

        Args:
        -----
            x : np.array

        Returns:
        --------
            Kx : np.array of same shape as input

        """
        return self._dirichlet(2*np.pi/self.period*x)

    def _dirichlet_derivative(self, x):
        x = np.squeeze(np.array(x))
        y = np.zeros(x.shape)
        indeterminate = np.mod(x, 2*np.pi) == 0
        y[indeterminate] = 0

        z = x[~indeterminate]
        a = self.n + .5
        b = .5
        num = a*np.cos(a*z)*np.sin(b*z) - b*np.cos(b*z)*np.sin(a*z)
        denom = np.power(np.sin(b*z), 2)
        y[~indeterminate] = np.divide(num, denom)
        return y

    def derivative(self, x):
        """Evaluate the derivative of the kernel.

        Args:
        -----
            x : np.array

        Returns:
        --------
            Kpx : np.array of same shape as input

        """
        return self._dirichlet_derivative(2*np.pi/self.period*x)


class GaussianKernel(BaseKernel):
    """Implement the Gaussian kernel."""

    def __init__(self, sigma):
        """Init.

        Args:
        -----
            sigma : float
                Width of the gaussian kernel.
        """
        self.sigma = sigma

    def state_dict(self):
        """Create a dict storing parameters."""
        return {
            'sigma': self.sigma,
        }

    def evaluate(self, x):
        """Gaussian kernel.

        Args:
            x : float or np.array

        Returns:
        --------
            float or np.array

        """
        x = np.squeeze(x)
        norm = (self.sigma*np.sqrt(2*np.pi))
        return np.exp(-.5*np.power(x, 2)/self.sigma**2)/norm

    def derivative(self, x):
        """Evaluate the derivative of the Gaussian kernel.

        Args:
            x : float or np.array

        Returns:
        --------
            float or np.array

        """
        x = np.squeeze(x)
        return -2*x*self.evaluate(x)/self.sigma**2
