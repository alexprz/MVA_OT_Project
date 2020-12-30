"""Implement activation functions."""
import numpy as np


def get_sigma(name):
    """Retrieve an activation function and its derivative from a name.

    Args:
    -----
        name : str

    Returns:
    --------
        callable
            Activation function
        callable
            Derivative of the activation function

    """
    sigmas = {
        'relu': (relu, relu_d),
        'sigmoid': (sigmoid, sigmoid_d),
    }
    return sigmas[name]


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
