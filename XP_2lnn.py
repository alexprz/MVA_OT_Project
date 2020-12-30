"""Implement the one hidden layer experiment of the paper."""
import numpy as np
import matplotlib.pyplot as plt

import two_layers_nn as tln
import optimizer as opt


np.random.seed(0)
m0 = 4
NNenv = tln.paper_env(m0, sigma_name='relu', loss_name='squared')

m = 10
eps = 1e-1
w0 = eps*np.ones(m)
roots = np.array([np.exp(2*np.pi*1j*k/m) for k in range(m)])[:, None]
theta0 = np.concatenate((np.real(roots), np.imag(roots)), axis=1)


bs = 5
n_iter = 10000
gamma0 = 1e-2
w, theta = opt.SGD(NNenv, w0, theta0, bs, n_iter, gamma0, print_every=100)

print(theta)

plt.scatter(w0*theta0[:, 0], w0*theta0[:, 1], color='blue', marker='.')
plt.scatter(w*theta[:, 0], w*theta[:, 1], color='red', marker='.')
plt.show()
