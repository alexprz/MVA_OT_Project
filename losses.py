"""Implement the loss functions for the NN example."""
import numpy as np

import activations as act


def quadratic(y1, y2):
    """Implement the quadratic loss.

    Args:
    -----
        y1 : np.array of shape (n,)
        y2 : np.array of shape (n,)

    Returns:
    --------
        l : np.array of shape (n,)

    """
    return np.power(y1 - y2, 2)/2


def quadratic_d1(y1, y2):
    """Implement the first derivative of the quadratic loss.

    Args:
    -----
        y1 : np.array of shape (n,)
        y2 : np.array of shape (n,)

    Returns:
    --------
        l : np.array of shape (n,)

    """
    return y1 - y2


def sigmoid(y1, y2):
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


def sigmoid_d1(y1, y2):
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
