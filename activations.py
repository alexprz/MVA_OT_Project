"""Implement activation functions."""
import numpy as np


def sigmoid(s):
    """Implement the sigmoid activation function.

    Args:
    -----
        s : np.array of shape (n,)

    Returns:
    --------
        sigma : np.array of shape (n,)

    """
    return np.divide(1, 1 + np.exp(-s))


def sigmoid_d(s):
    """Implement the derivative of the sigmoid activation function.

    Args:
    -----
        s : np.array of shape (n,)

    Returns:
    --------
        sigma : np.array of shape (n,)

    """
    sigm = sigmoid(s)
    return sigm*(1 - sigm)


def relu(s):
    """Implement the ReLU activation function.

    Args:
    -----
        s : np.array of shape (n,)

    Returns:
    --------
        r : np.array of shape (n,)

    """
    return np.maximum(0, s)


def relu_d(s):
    """Implement the "derivative" of the ReLU activation function.

    Args:
    -----
        s : np.array of shape (n,)

    Returns:
    --------
        r : np.array of shape (n,)

    """
    return np.array(s > 0).astype(int)
