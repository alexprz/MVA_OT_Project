"""Implement the two layer neural network example of the paper."""
import numpy as np

import losses
import activations


class TwoLayerNN():
    """Implement functions for the NN example of the paper."""

    def __init__(self, activation, loss, w_bar, theta_bar):
        """Init.

        Args:
        -----
            activation : activations.BaseActivation object
            loss : losses.BaseLoss object
            w_bar : np.array of shape (m,)
                The optimal w
            theta_bar : np.array of shape (m, d)
                The optimal theta

        """
        # Check arguments
        assert isinstance(activation, activations.BaseActivation)
        assert isinstance(loss, losses.BaseLoss)

        self.activation = activation
        self.loss = loss

        self.w_bar = w_bar
        self.theta_bar = theta_bar

    @property
    def d(self):
        """Dimension of the input samples."""
        return self.theta_bar.shape[1] - 1

    def layer1(self, theta, x, use_derivative=False):
        """Implement the first layer (the hidden layer) of the paper.

        Args:
        -----
            theta : np.array of shape (m, d)
            x : np.array of shape (n, d-1)
            use_derivative : bool
                Whether to use the derivative of the activation function

        Returns:
        --------
            np.array of shape (n, m)

        """
        act = self.activation.derivative if use_derivative else self.activation
        return act(np.inner(x, theta[:, :-1]) + theta[None, :, -1])

    def layer2(self, w, theta, x):
        """Implement the second layer of the paper.

        Args:
        -----
            w : np.array of shape (m,)
            theta : np.array of shape (m, d)
            x : np.array of shape (n, d-1)

        Returns:
        --------
            np.array of shape (n,)

        """
        return np.mean(self.phi(w, theta, x), axis=1)

    def phi(self, w, theta, x):
        """Implement the phi function of the paper.

        Args:
        -----
            w : np.array of shape (m,)
            theta : np.array of shape (m, d)
            x : np.array of shape (n, d-1)

        Returns:
        --------
            np.array of shape (n, m)

        """
        return w[None, :]*self.layer1(theta, x)  # (n, m)

    def phi_dw(self, theta, x):
        """Implement the derivative of the phi function wrt w.

        Args:
        -----
            theta : np.array of shape (m, d)
            x : np.array of shape (n, d-1)

        Returns:
        --------
            np.array of shape (n, m)

        """
        return self.layer1(theta, x)  # (n, m)

    def _phi_dtheta1(self, w, theta, x):
        """Implement the derivative of phi wrt the first d-1 thetas.

        Args:
        -----
            w : np.array of shape (m,)
            theta : np.array of shape (m, d)
            x : np.array of shape (n, d-1)

        Returns:
        --------
            np.array of shape (n, m, d-1)

        """
        layer1 = self.layer1(theta, x, use_derivative=True)  # (n, m)
        return w[None, :, None]*x[:, None, :]*layer1[:, :, None]

    def _phi_dtheta2(self, w, theta, x):
        """Implement the derivative of the phi function wrt the last theta.

        Args:
        -----
            w : np.array of shape (m,)
            theta : np.array of shape (m, d)
            x : np.array of shape (n, d-1)

        Returns:
        --------
            np.array of shape (n, m)

        """
        layer1 = self.layer1(theta, x, use_derivative=True)
        return w[None, :]*layer1  # (n, m)

    def phi_dtheta(self, w, theta, x):
        """Implement the derivative of the phi function wrt theta.

        Args:
        -----
            w : np.array of shape (m,)
            theta : np.array of shape (m, d)
            x : np.array of shape (n, d-1)

        Returns:
        --------
            np.array of shape (n, m)

        """
        d1 = self._phi_dtheta1(w, theta, x)
        d2 = self._phi_dtheta2(w, theta, x)[:, :, None]
        return np.concatenate((d1, d2), axis=2)

    def y(self, w, theta, x):
        """Implement the output function.

        Args:
        -----
            w : np.array of shape (m,)
            theta : np.array of shape (m, d)
            x : np.array of shape (n, d-1)

        Returns:
        --------
            np.array of shape (n,)

        """
        return self.layer2(w, theta, x)

    def y_bar(self, x):
        """Implement the optimal output function.

        Args:
        -----
            x : np.array of shape (n, d-1)

        Returns:
        --------
            np.array of shape (n,)

        """
        return self.layer2(self.w_bar, self.theta_bar, x)

    # @staticmethod
    # def V(w, theta):
    #     """Implement the second layer of the paper.

    #     Args:
    #     -----
    #         w : np.array of shape (m,)
    #         theta : np.array of shape (m, d)

    #     Returns:
    #     --------
    #         np.array of shape (m,)

    #     """
    #     return np.abs(w)*np.linalg.norm(theta, ord=1, axis=1)

    @staticmethod
    def V(w, theta):
        """Implement the second layer of the paper.

        Args:
        -----
            w : np.array of shape (m,)
            theta : np.array of shape (m, d)

        Returns:
        --------
            np.array of shape (m,)

        """
        return np.abs(w)

    @staticmethod
    def V_dw(w, theta):
        """Implement the second layer of the paper.

        Args:
        -----
            w : np.array of shape (m,)
            theta : np.array of shape (m, d)

        Returns:
        --------
            np.array of shape (m,)

        """
        return np.sign(w)

    @staticmethod
    def V_dtheta(w, theta):
        """Implement the second layer of the paper.

        Args:
        -----
            w : np.array of shape (m,)
            theta : np.array of shape (m, d)

        Returns:
        --------
            np.array of shape (m, d)

        """
        return np.zeros_like(theta)

    @staticmethod
    def Vm(w, theta):
        """Implement the second layer of the paper.

        Args:
        -----
            w : np.array of shape (m,)
            theta : np.array of shape (m, d)

        Returns:
        --------
            float

        """
        return TwoLayerNN.V(w, theta).mean().item()

    @staticmethod
    def subgrad_Vm(w, theta):
        """Return a subgradient of V.

        Args:
        -----
            w : np.array of shape (m,)
            theta : np.array of shape (m,)

        Returns:
        --------
            subgrad_w : np.array of shape (m,)
            subgrad_theta : np.array of shape (m, d)

        """
        m = w.shape[0]
        subgrad_w = TwoLayerNN.V_dw(w, theta)
        subgrad_theta = TwoLayerNN.V_dtheta(w, theta)

        return subgrad_w/m, subgrad_theta/m

    def Rm(self, w, theta, x):
        """Gradient of R wrt w and theta.

        Args:
        -----
            w : np.array of shape (m,)
            theta : np.array of shape (m, d)
            x : np.array of shape (n, d-1)

        Returns:
        --------
            float

        """
        y_hat = self.y(w, theta, x)
        y_bar = self.y_bar(x)
        return self.loss(y_hat, y_bar).mean().item()

    def grad_Rm(self, w, theta, x):
        """Gradient of R wrt w and theta.

        Args:
        -----
            w : np.array of shape (m,)
            theta : np.array of shape (m, d)
            x : np.array of shape (n, d-1)

        Returns:
        --------
            grad_w : np.array of shape (m,)
            grad_theta : np.array of shape (m, d)

        """
        m = w.shape[0]
        y_hat = self.y(w, theta, x)
        y_bar = self.y_bar(x)
        loss_d = self.loss.derivative(y_hat, y_bar)

        grad_w = self.phi_dw(theta, x)*loss_d[:, None]/m
        grad_theta = self.phi_dtheta(w, theta, x)*loss_d[:, None, None]/m

        return grad_w.mean(axis=0), grad_theta.mean(axis=0)

    def fm(self, w, theta, x):
        """Implement the discretized objective function F.

        Args:
        -----
            w : np.array of shape (m,)
            theta : np.array of shape (m, d)
            x : np.array of shape (n, d-1)

        Returns:
        --------
            float

        """
        return self.Rm(w, theta, x) + self.Vm(w, theta)

    def grad_fm(self, w, theta, x):
        """Implement the gradient of the discretized objective function F.

        Args:
        -----
            w : np.array of shape (m,)
            theta : np.array of shape (m, d)
            x : np.array of shape (n, d-1)

        Returns:
        --------
            grad_w : np.array of shape (m,)
            grad_theta : np.array of shape (m, d)

        """
        grad_w_r, grad_theta_r = self.grad_Rm(w, theta, x)
        grad_w_v, grad_theta_v = self.subgrad_Vm(w, theta)

        return grad_w_r + grad_w_v, grad_theta_r + grad_theta_v


def paper_env(m0, activation, loss):
    """Create the environment of the paper.

    Args:
    -----
        m0 : int
            Number of neurons in the ground truth
        activation : activations.BaseActivation object
        loss : losses.BaseLoss object

    Returns:
    --------
        two_layer_nn.TwoLayerNN object

    """
    d = 3

    # Generate ground truth
    w_bar = np.random.normal(0, 1, size=m0)
    mean, cov = np.zeros(d-1), np.eye(d-1)
    theta_bar = np.random.multivariate_normal(mean, cov, size=m0)

    return TwoLayerNN(activation, loss, w_bar, theta_bar)
