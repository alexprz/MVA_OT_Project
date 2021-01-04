"""Implement activation functions."""
import numpy as np
from abc import ABC, abstractmethod


class BaseActivation(ABC):
    """Abstract class for activation classes."""

    def __call__(self, s):
        """Shortcut to call the activation function."""
        return self.activation(s)

    def __str__(self):
        """Give a str representation of the parameters."""
        return ''.join([f'-{k}_{v}' for k, v in self.state_dict().items()])

    @staticmethod
    @abstractmethod
    def activation(s):
        """Implement the activation function."""
        return

    @staticmethod
    @abstractmethod
    def derivative(s):
        """Implement the derivative of the activation function."""
        return

    @abstractmethod
    def state_dict(self):
        """Create a dict storing parameters."""
        return


class Sigmoid(BaseActivation):
    """Implement the sigmoid activation function."""

    @staticmethod
    def activation(s):
        """Implement the sigmoid activation function.

        Args:
        -----
            s : np.array of shape (n,)

        Returns:
        --------
            sigma : np.array of shape (n,)

        """
        return np.power(1 + np.exp(-s), -1)

    @staticmethod
    def derivative(s):
        """Implement the derivative of the sigmoid activation function.

        Args:
        -----
            s : np.array of shape (n,)

        Returns:
        --------
            sigma : np.array of shape (n,)

        """
        sigmoid = Sigmoid.activation(s)
        return sigmoid*(1 - sigmoid)

    def state_dict(self):
        """Create a dict storing parameters."""
        return {'name': 'sigmoid'}


class ReLU(BaseActivation):
    """Implement the ReLU activation function."""

    @staticmethod
    def activation(s):
        """Implement the ReLU activation function.

        Args:
        -----
            s : np.array of shape (n,)

        Returns:
        --------
            r : np.array of shape (n,)

        """
        return np.maximum(0, s)

    @staticmethod
    def derivative(s):
        """Implement the "derivative" of the ReLU activation function.

        Args:
        -----
            s : np.array of shape (n,)

        Returns:
        --------
            r : np.array of shape (n,)

        """
        return np.array(s > 0).astype(int)

    def state_dict(self):
        """Create a dict storing parameters."""
        return {'name': 'relu'}
