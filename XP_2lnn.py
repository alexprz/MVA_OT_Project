"""Implement the one hidden layer experiment of the paper."""
import numpy as np
import matplotlib.pyplot as plt

import two_layers_nn as tln
import optimizer as opt


np.random.seed(0)
m0 = 4
NNenv = tln.paper_env(m0, sigma_name='relu', loss_name='squared')

m = 20
eps = 1e-1
w0 = eps*np.ones(m)
roots = np.array([np.exp(2*np.pi*1j*k/m) for k in range(m)])[:, None]
theta0 = np.concatenate((np.real(roots), np.imag(roots)), axis=1)


bs = 2
n_iter = 10000
gamma0 = 1e0
ws, thetas = opt.SGD(NNenv, w0, theta0, bs, n_iter, gamma0, print_every=100)

w_final, theta_final = ws[-1, ...], thetas[-1, ...]


# Plot ground truth
XX = np.linspace(0, 2, 2)

theta_bar = NNenv.theta_bar
for k in range(m0):
    plt.plot(np.sign(theta_bar[k, 0])*XX, theta_bar[k, 1]/theta_bar[k, 0]*XX, linestyle='--', color='black')

plt.scatter(w0*theta0[:, 0], w0*theta0[:, 1], color='blue', marker='.')
plt.scatter(w_final*theta_final[:, 0], w_final*theta_final[:, 1], color='red', marker='.')

# Plot paths
for k in range(m):
    label = 'Flow' if k == 0 else ''
    plt.plot(ws[:, k]*thetas[:, k, 0], ws[:, k]*thetas[:, k, 1], color='green', linewidth=1, label=label)#, marker='o', markersize=1)

# y_min, y_max = ax.get_ylim()
# max_ylim = max(abs(y_min), abs(y_max))
ax = plt.gca()
ax.set_ylim(-2, 2)

plt.legend()
plt.show()
