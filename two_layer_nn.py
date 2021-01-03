"""Implement the two layer neural network example of the paper."""
import numpy as np


class TwoLayerNN():
    """Implement functions for the NN example of the paper."""

    def __init__(self, params):
        """Init.

        Args:
        -----
            params : parameters.TLNParameters object

        """
        self.activation = params.activation
        self.loss = params.loss

        self.w_bar = params.w_bar
        self.theta_bar = params.theta_bar

        self.beta = 1/params.lbd
        self.params = params

        # Estimate the mean value of the ground truth y_bar
        mean, cov = np.zeros(self.d), np.eye(self.d)
        x = np.random.multivariate_normal(mean, cov, size=1000)
        self.y_bar_mean = self.layer2(self.w_bar, self.theta_bar, x).mean()

    @property
    def d(self):
        """Dimension of the input samples."""
        # return self.theta_bar.shape[1] - 1
        return self.theta_bar.shape[1]

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
        # x = x.reshape(-1, 1) if x.ndim == 1 else x
        # x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
        return act(np.inner(x, theta))

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
            np.array of shape (n, m, d)

        """
        d1 = self._phi_dtheta1(w, theta, x)
        # d2 = self._phi_dtheta2(w, theta, x)[:, :, None]

        # print('d1', d1.shape)
        # print('d2', d2.shape)
        # return np.concatenate((d1, d2), axis=2)
        return d1

    def to_class(self, y):
        return y
        # y_class = np.copy(y)
        # y_class[y >= self.y_bar_mean] = 1
        # y_class[y < self.y_bar_mean] = -1
        # return y_class

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
        return self.to_class(self.layer2(w, theta, x))

    def y_bar(self, x):
        """Implement the optimal output function.

        Args:
        -----
            x : np.array of shape (n, d-1)

        Returns:
        --------
            np.array of shape (n,)

        """
        # return self.layer2(self.w_bar, self.theta_bar, x)
        return self.y(self.w_bar, self.theta_bar, x)

    # def Vm(self, w, theta):
    #     """Implement the second layer of the paper.

    #     Args:
    #     -----
    #         w : np.array of shape (m,)
    #         theta : np.array of shape (m, d)

    #     Returns:
    #     --------
    #         float

    #     """
    #     return self.beta*np.abs(w).mean().item()

    # def subgrad_Vm(self, w, theta):
    #     """Return a subgradient of V.

    #     Args:
    #     -----
    #         w : np.array of shape (m,)
    #         theta : np.array of shape (m,)

    #     Returns:
    #     --------
    #         subgrad_w : np.array of shape (m,)
    #         subgrad_theta : np.array of shape (m, d)

    #     """
    #     m = w.shape[0]
    #     subgrad_w = self.beta*np.sign(w)/m
    #     subgrad_theta = self.beta*np.zeros_like(theta)/m

    #     return subgrad_w, subgrad_theta

    def Vm(self, w, theta):
        """Implement the second layer of the paper.

        Args:
        -----
            w : np.array of shape (m,)
            theta : np.array of shape (m, d)

        Returns:
        --------
            float

        """
        m = theta.shape[0]
        return self.beta*(np.linalg.norm(w)**2 + np.linalg.norm(theta)**2)/m

    def subgrad_Vm(self, w, theta):
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
        subgrad_w = 2*self.beta*w/m
        subgrad_theta = 2*self.beta*theta/m

        return subgrad_w, subgrad_theta

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
        loss_d = self.loss.derivative(y_hat, y_bar)  # (n,)

        grad_w = self.phi_dw(theta, x)*loss_d[:, None]/m  # (n, m)
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


def paper_ground_truth(m0):
    """Create the environment of the paper.

    Args:
    -----
        m0 : int
            Number of neurons in the ground truth

    Returns:
    --------
            w_bar : np.array of shape (m,)
            theta_bar : np.array of shape (m, 2)

    """
    d = 2

    # Generate ground truth
    w_bar = np.random.normal(0, 1, size=m0)
    mean, cov = np.zeros(d), np.eye(d)
    theta_bar = np.random.multivariate_normal(mean, cov, size=m0)

    return w_bar, theta_bar


def paper2_ground_truth(m0):
    """Create the environment of the second paper.

    Args:
    -----
        m0 : int
            Number of neurons in the ground truth

    Returns:
    --------
            w_bar : np.array of shape (m,)
            theta_bar : np.array of shape (m, 2)

    """
    d = 2

    # Generate ground truth
    w_bar = np.sign(np.random.normal(0, 1, size=m0))
    theta_bar = np.random.normal(0, 1, size=(m0, d))


    # Normalize
    theta_bar /= np.linalg.norm(theta_bar, axis=1)[:, None]

    return w_bar, theta_bar
