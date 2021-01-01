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


bs = 100
n_iter = 1000
gamma0 = 1e-1
ws, thetas, losses, losses_d = opt.SGD(NNenv, w0, theta0, bs, n_iter, gamma0, print_every=10)

w_final, theta_final = ws[-1, ...], thetas[-1, ...]


# Plot loss evolution
mean, cov = np.zeros(NNenv.d), np.eye(NNenv.d)
x = np.random.multivariate_normal(mean, cov, size=1000)
y = NNenv.y(x)
y_hat = NNenv.forward(NNenv.w_bar, NNenv.theta_bar, x)
optimal_loss = NNenv.loss(y_hat, y).mean()
optimal_loss_d = NNenv.loss_d1(y_hat, y).mean()

fig = plt.figure(figsize=(16, 6))
ax = fig.add_subplot(121)
ax.plot(losses)
print(f'Optimal loss: {optimal_loss}')
print(f'Grad optimal loss: {optimal_loss_d}')
ax.axhline(optimal_loss, linestyle=':', color='black', label=f'Optimal loss ({optimal_loss:.2e})')

ax.legend()

# Plot ground truth
ax2 = fig.add_subplot(122)
xx = np.linspace(0, 2, 2)

theta_bar = NNenv.theta_bar
XX = np.sign(theta_bar[:, 0])[:, None]*xx[None, :]
YY = np.divide(theta_bar[:, 1], theta_bar[:, 0])[:, None]*xx[None, :]
XX = np.clip(XX, -2, 2)
YY = np.clip(YY, -2, 2)
for k in range(m0):
    # plt.plot(np.sign(theta_bar[k, 0])*XX, theta_bar[k, 1]/theta_bar[k, 0]*XX, linestyle='--', color='black')
    ax2.plot(XX[k, :], YY[k, :], linestyle='--', color='black')
    ax2.scatter(w0[k]*XX[k, -1], w0[k]*YY[k, -1], marker='+', color='orange')

ax2.scatter(w0*theta0[:, 0], w0*theta0[:, 1], color='blue', marker='.')
ax2.scatter(w_final*theta_final[:, 0], w_final*theta_final[:, 1], color='red', marker='.')

# Plot paths
for k in range(m):
    label = 'Flow' if k == 0 else ''
    ax2.plot(ws[:, k]*thetas[:, k, 0], ws[:, k]*thetas[:, k, 1], color='green', linewidth=1, label=label)#, marker='o', markersize=1)

# y_min, y_max = ax.get_ylim()
# max_ylim = max(abs(y_min), abs(y_max))
# ax = plt.gca()
# ax.set_ylim(-2, 2)

ax2.legend()
plt.show()
