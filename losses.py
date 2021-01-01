"""Implement the loss functions for the NN example."""
import numpy as np

import activations as act


def get_loss(name):
    """Retrieve a loss function and its derivative from a name.

    Args:
    -----
        name : str

    Returns:
    --------
        callable
            Loss function
        callable
            Derivative of the loss function

    """
    losses = {
        'squared': Squared(),
        'logistic': Logistic(),
    }
    return losses[name]


class Squared():
    """Implement the squared loss function."""

    def __call__(self, y1, y2):
        """Shortcut to call the loss function."""
        return self.loss(y1, y2)

    @staticmethod
    def loss(y1, y2):
        """Implement the squared loss.

        Args:
        -----
            y1 : np.array of shape (n,)
            y2 : np.array of shape (n,)

        Returns:
        --------
            l : np.array of shape (n,)

        """
        return np.power(y1 - y2, 2)/2

    @staticmethod
    def derivative(y1, y2):
        """Implement the first derivative of the squared loss.

        Args:
        -----
            y1 : np.array of shape (n,)
            y2 : np.array of shape (n,)

        Returns:
        --------
            l : np.array of shape (n,)

        """
        return y1 - y2


class Logistic():
    """Implement the logistic loss function."""

    def __call__(self, y1, y2):
        """Shortcut to call the loss function."""
        return self.loss(y1, y2)

    @staticmethod
    def loss(y1, y2):
        """Implement the logistic loss.

        Args:
        -----
            y1 : np.array of shape (n,)
            y2 : np.array of shape (n,)

        Returns:
        --------
            l : np.array of shape (n,)

        """
        sigmoid = act.Sigmoid.activation
        return -np.log(sigmoid(y1*y2))/np.log(2)

    @staticmethod
    def derivative(y1, y2):
        """Implement the first derivative of the logistic loss.

        Args:
        -----
            y1 : np.array of shape (n,)
            y2 : np.array of shape (n,)

        Returns:
        --------
            l : np.array of shape (n,)

        """
        sigmoid_d = act.Sigmoid.derivative
        return np.divide(y2, sigmoid_d(y1*y2))/np.log(2)
