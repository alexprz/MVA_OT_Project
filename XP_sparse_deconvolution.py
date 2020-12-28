"""Implement the sparse deconvolution experience of the paper."""
import os
import numpy as np
import matplotlib.pyplot as plt

import sparse_deconvolution_1D as sd1
import optimizer as opt


plt.rcParams.update({
    'text.usetex': True,
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
    'font.size': 15,
    'figure.figsize': (12, 6),
})

np.random.seed(0)

# Retrieve the environment of the paper
m0 = 5
env = sd1.paper_env(m0)

ax = plt.gca()

# Print the optimal positions
for i, p in enumerate(env.p):
    sign = np.sign(env.w[i])
    ymin, ymax = (0.5, 1) if sign > 0 else (0, 0.5)
    label = 'Optimal positions' if i == 0 else ''
    ax.axvline(p, ymin=ymin, ymax=ymax, color='black', linestyle='--',
               linewidth=1, label=label)

# Initialize the particles
m = 20  # Number of particles
w0 = np.zeros(m)  # Weights
theta0 = np.arange(m)/m  # Positions (uniformly spaced)
n = 50  # Discretization (for integral computation)
max_iter = 1000000

# Plot initial particles
plt.scatter(theta0, w0, color='blue', marker='.')

# Apply the forward backward algorithm
ws, thetas = opt.forward_backward(env, w0, theta0, max_iter=max_iter, n=n,
                                  print_every=100)

# Plot the final particles
plt.scatter(thetas[-1, :], ws[-1, :], color='red', marker='.', label='Particle')

# Plot the particles' trajectories during optimization
for k in range(m):
    label = 'Flow' if k == 0 else ''
    plt.plot(thetas[:, k], ws[:, k], color='green', linewidth=0.8, label=label)

# Center the plot on the y axis
y_min, y_max = ax.get_ylim()
max_ylim = max(abs(y_min), abs(y_max))
ax.set_ylim(-max_ylim, max_ylim)

# Dump
folder = 'fig/dump/sd1_paper/'
os.makedirs(folder, exist_ok=True)
ws.dump(folder+'ws.pickle')
thetas.dump(folder+'thetas.pickle')

# Plot
plt.legend()
plt.show()
