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
        'squared': (squared, squared_d1),
        'logistic': (logistic, logistic_d1),
    }
    return losses[name]


def squared(y1, y2):
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


def squared_d1(y1, y2):
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


def logistic(y1, y2):
    """Implement the logistic loss.

    Args:
    -----
        y1 : np.array of shape (n,)
        y2 : np.array of shape (n,)

    Returns:
    --------
        l : np.array of shape (n,)

    """
    return -np.log(act.sigmoid(y1*y2))/np.log(2)


def logistic_d1(y1, y2):
    """Implement the first derivative of the logistic loss.

    Args:
    -----
        y1 : np.array of shape (n,)
        y2 : np.array of shape (n,)

    Returns:
    --------
        l : np.array of shape (n,)

    """
    return np.divide(y2, act.sigmoid_d(y1*y2))/np.log(2)
