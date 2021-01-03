"""Implement the sparse deconvolution example of the paper."""
import numpy as np



class SparseDeconvolution():
    """Implement the sparse deconvolution example of the paper."""

    def __init__(self, params):
        """Init.

        Args:
        -----
            w_bar : np.array of shape (m,)
                The ground truth weights
            theta_bar : np.array of shape (m,)
                The ground truth positions
            params : parameters.SD1Parameters object

        """
        self.params = params
        self.kernel = params.kernel
        self.w_bar = params.w_bar
        self.theta_bar = params.theta_bar
        self.lbd = params.lbd

        self.x_min = 0
        self.x_max = 1

# def spikes_1D(x, w, p):
#     """1D weighted spikes.

#     Args:
#     -----
#         x : float
#         w : array of shape (m,)
#             Weights of the spikes.
#         p : array of shape (m,)
#             Positions of the spikes.

#     Returns:
#     --------
#         float

#     """
#     assert w.shape == p.shape
#     assert len(w.shape) == 1

#     return np.sum(w[p == x])

    def discretize(self):
        """Discretize space to compute the inegral.

        Args:
        -----
            n : int
                Number of points

        Returns:
        --------
            np.array of shape (n,)

        """
        return np.linspace(self.x_min, self.x_max, self.params.n)

    @staticmethod
    def draw_positions_1D(m):
        """Draw positions in (0, 1) with a minimum separaton of 0.1.

        Args:
        -----
            m : int
                Number of positions to draw.

        Returns:
        --------
            p : array of shape (m,)

        """
        odd = np.random.binomial(n=1, p=0.5)
        chunks = np.random.choice(np.arange(odd, 10, 2), size=m, replace=False)
        return (chunks + np.random.uniform(0, 1, size=m))/10

    def y(self, x, w, theta):
        """Implement the y function with given weigths and positions.

        Args:
        -----
            x : np.array of shape (N,) or scalar
            w : np.array of shape (m,)
            theta : np.array of shape (m,)

        Returns:
        --------
            y : np.array of shape (N,)

        """
        x = np.squeeze(np.array(x))
        _y = np.dot(self.kernel(x[:, None] - theta[None, :]), w)
        return _y

    def y_bar(self, x):
        """Implement the signal y function with ground truth weigths and pos.

        Args:
        -----
            x : np.array of shape (N,) or scalar

        Returns:
        --------
            y : np.array of shape (N,)

        """
        return self.y(x, self.w_bar, self.theta_bar)

    def R(self, f):
        """Implement the loss function.

        Args:
        -----
            f : np.array of shape (d, N)
                N values of d functions f on (0, 1) equally spaced

        Returns:
        --------
            float

        """
        f = np.array(f)
        f = f.reshape(1, -1) if len(f.shape) == 1 else f
        N = f.shape[1]
        linspace = np.linspace(0, 1, N)
        return 1/(2*self.lbd*N)*np.sum(np.power(f - self.y_bar(linspace)[None, :], 2), axis=1)

    def grad_R(self, w, theta, x):
        """Gradient of R wrt w and theta.

        Args:
        -----
            w : np.array of shape (m,)
            theta : np.array of shape (m,)
            x : np.array of shape (n,)

        Returns:
        --------
            grad_w : np.array of shape (m,)
            grad_theta : np.array of shape (m,)

        """
        m = w.shape[0]
        theta = np.squeeze(theta)
        w = np.squeeze(w)
        x = np.squeeze(x)
        dx = x[:, None] - theta[None, :]  # (n, m)
        psi_t = self.kernel(dx)  # (n, m)
        z = (self.y_bar(x) - np.dot(psi_t, w)/m)  # (n,)
        grad_w = -np.mean(psi_t*z[:, None], axis=0)/(self.lbd*m)  # (m,)
        grad_theta = w*np.mean(self.kernel.derivative(dx)*z[:, None], axis=0)/(self.lbd*m)  # (m,)

        return grad_w, grad_theta

    @staticmethod
    def V(w, theta):
        return np.abs(w)

    @staticmethod
    def subgrad_V(w, theta):
        """Return a subgradient of V.

        Args:
        -----
            w : np.array of shape (m,)
            theta : np.array of shape (m,)

        Returns:
        --------
            subgrad_w : np.array of shape (m,)
            subgrad_theta : np.array of shape (m,)

        """
        m = w.shape[0]
        subgrad_w = np.sign(w)/m
        subgrad_theta = np.zeros_like(theta)

        return subgrad_w, subgrad_theta

    @staticmethod
    def prox_V(w, theta, gamma):
        """Compute the proximity operator of gamma*V.

        Args:
        -----
            w : np.array of shape (m,)
            theta : np.array of shape (m,)
            gamma : float

        Returns:
            prox_w : np.array of shape (m,)
            prox_theta : np.array of shape (m,)

        """
        m = w.shape[0]
        prox_w = np.sign(w)*np.maximum(np.abs(w) - gamma/m, 0)
        prox_theta = theta

        return prox_w, prox_theta

    def phi(self, w, theta, x):
        """Implement the phi function.

        Args:
        -----
            w : np.array of shape (m,)
            theta : np.array of shape (m, d)
            x : np.array of shape (N, d)

        Returns:
        --------
            np.array of shape (m, N)

        """
        x = x.reshape(-1, 1) if x.ndim == 1 else x
        theta = theta.reshape(-1, 1) if theta.ndim == 1 else theta
        w = np.squeeze(w)
        return w[:, None]*self.kernel(x[None, :, :] - theta[:, None, :])

    def f_m(self, w, theta, n=1000):
        """Implement the discretized objective function F.

        Args:
        -----
            env : Env named tuple
            w : np.array of shape (m,)
            theta : np.array of shape (m, d)
            n : int
                Discretization for the integral computation

        Returns:
        --------
            float

        """
        x = np.linspace(self.x_min, self.x_max, n)

        fm = self.R(self.phi(w, theta, x).mean(axis=0)) + self.V(w, theta).mean()
        return fm.item()

    def subgrad_f_m(self, w, theta, n=1000):
        """Evaluate a subgradient of the objective f_m.

        Args:
        -----
            w : np.array of shape (m,)
            theta : np.array of shape (m, d)
            n : int
                Discretization for the integral computation

        Returns:
        --------
            subgrad_w : np.array of shape (m,)
            subgrad_theta : np.array of shape (m, d)

        """
        x = np.linspace(self.x_min, self.x_max, n)
        grad_R = self.grad_R(w, theta, x)
        subgrad_V = self.subgrad_V(w, theta)
        return grad_R[0] + subgrad_V[0], grad_R[1] + subgrad_V[1]


def paper_ground_truth(m0):
    """Create a sparse deconvolution environment from parameters.

    Args:
    -----
        params : parameters.SD1Parameters object

    Returns:
    --------
        SparseDeconvolution object

    """
    signs = 2*np.random.binomial(n=1, p=0.5, size=m0)-1  # weight signs
    w_bar = signs*np.random.uniform(0.5, 1.5, size=m0)  # weights
    theta_bar = SparseDeconvolution.draw_positions_1D(m0)  # positions

    return w_bar, theta_bar
