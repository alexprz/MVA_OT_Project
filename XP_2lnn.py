"""Implement the one hidden layer experiment of the paper."""
import os
import time
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib import cm

import two_layer_nn as tln
import optimizer as opt
import activations as act
import losses


plt.rcParams.update({
    'text.usetex': True,
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
    'font.size': 10,
    'figure.figsize': (12, 6),
})


def scatterplot(w, theta, ax, **kwargs):
    """Scatter particles."""
    x = w*theta[:, 0]
    y = w*theta[:, 1]
    ax.scatter(x, y, **kwargs)


def lineplot(w, theta, ax, **kwargs):
    """Draw lines between (0, 0) and particles."""
    x = np.stack((np.zeros_like(w), w*theta[:, 0]), axis=0)
    y = np.stack((np.zeros_like(w), w*theta[:, 1]), axis=0)
    # x = np.stack((-w*theta[:, 0], w*theta[:, 0]), axis=0)
    # y = np.stack((-w*theta[:, 1], w*theta[:, 1]), axis=0)
    ax.plot(1e1*x, 1e1*y, **kwargs)


np.random.seed(0)
m0 = 5
# tln_env = tln.paper_env(m0, act.ReLU(), losses.Squared(), beta=1)
tln_env = tln.his_code_env(m0, lbd=3e-2)

# Initialize the particle flow
m = 100
eps = 1e-1
w0 = eps*np.ones(m)
roots = np.array([np.exp(2*np.pi*1j*k/m) for k in range(m)])[:, None]
theta0 = np.concatenate((np.real(roots), np.imag(roots)), axis=1)
w_bar, theta_bar = tln_env.w_bar, tln_env.theta_bar

# u0 = np.random.normal(0, 1, size=(m, tln_env.d+1))
# u0 /= np.linalg.norm(u0, axis=1)[:, None]
# w0 = u0[:, 0]
# theta0 = u0[:, 1:]

print(w0.shape)
print(theta0.shape)

# w_bar = np.zeros_like(w0)
# theta_bar = np.zeros_like(theta0)
# w_bar[:m0] = m/m0*tln_env.w_bar
# theta_bar[:m0, :] = tln_env.theta_bar
# w0 = np.copy(w_bar)
# theta0 = np.copy(theta_bar)

# Optimize the particle flow
bs = 15
n_iter = 1000
gamma0 = 2
ws, thetas, fms, norm_grad_fms, Rms, Vms, norm_grad_Rms, norm_grad_Vms = opt.SGD(tln_env, w0, theta0, bs, n_iter, gamma0, print_every=10)

w_final, theta_final = ws[-1, ...], thetas[-1, ...]

# Plot ground truth
fig = plt.figure(figsize=(18, 8))
ax = fig.add_subplot(231)
scatterplot(tln_env.w_bar, tln_env.theta_bar, ax, marker='+', color='orange')
ax.set_xlabel(r'$\theta_1$')
ax.set_ylabel(r'$\theta_2$')
ax.set_title('Trajectories of the particles')

# Plot particle paths and start/end
scatterplot(w0, theta0, ax, color='blue', marker='.')
scatterplot(w_final, theta_final, ax, color='red', marker='.')
for k in range(m):
    label = 'Flow' if k == 0 else ''
    ax.plot(ws[:, k]*thetas[:, k, 0], ws[:, k]*thetas[:, k, 1], color='green', linewidth=.5, label=label)#, marker='o', markersize=1)

# Plot lines of optimal positions
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()
lineplot(tln_env.w_bar, tln_env.theta_bar, ax, linestyle='--', color='black', label='Optimal positions')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.legend()
lineplot(w_final, theta_final, ax, linestyle=':', color='cyan')

# Plot objective evolution
mean, cov = np.zeros(tln_env.d), np.eye(tln_env.d)
x = np.random.multivariate_normal(mean, cov, size=1000)
fm_bar = tln_env.fm(w_bar, theta_bar, x)
Rm_bar = tln_env.Rm(w_bar, theta_bar, x)
Vm_bar = tln_env.Vm(w_bar, theta_bar)

ax = fig.add_subplot(232)
pF = ax.plot(fms, label='$F_m = R_m + V_m$')
pR = ax.plot(Rms, label='$R_m$', linewidth=.5)
pV = ax.plot(Vms, label='$V_m$', linewidth=.5)
print(f'Optimal fm: {fm_bar}')
ax.axhline(Rm_bar, linestyle=':', color=pR[0].get_color(), label=f'$\\bar{{R}}$ ({Rm_bar:.3e})')
ax.axhline(Vm_bar, linestyle=':', color=pV[0].get_color(), label=f'$\\bar{{V}}$ ({Vm_bar:.3e})')
ax.axhline(fm_bar, linestyle=':', color=pF[0].get_color(), label=f'$\\bar{{F}} = \\bar{{R}} + \\bar{{V}}$ ({fm_bar:.3e})')
ax.set_xlabel('Iterations')
ax.set_ylabel(r'$F_m$')
ax.set_title(r'Evolution of $F_m$')
ax.legend()

# Gradient
grad_w_fm, grad_theta_fm = tln_env.grad_fm(w_bar, theta_bar, x)
grad_w_Rm, grad_theta_Rm = tln_env.grad_Rm(w_bar, theta_bar, x)
grad_w_Vm, grad_theta_Vm = tln_env.subgrad_Vm(w_bar, theta_bar)
norm_grad_fm_bar = norm(grad_w_fm) + norm(grad_theta_fm)
norm_grad_Rm_bar = norm(grad_w_Rm) + norm(grad_theta_Rm)
norm_grad_Vm_bar = norm(grad_w_Vm) + norm(grad_theta_Vm)

ax = fig.add_subplot(233)
pF = ax.plot(norm_grad_fms, label='$\\|\\nabla F_m\\|_2 = \\|\\nabla R_m + \\nabla V_m\\|_2$')
pR = ax.plot(norm_grad_Rms, label='$\\|\\nabla R_m\\|_2$', linewidth=.5)
pV = ax.plot(norm_grad_Vms, label='$\\|\\nabla V_m\\|_2$', linewidth=.5)
print(f'Gradient optimal fm: {norm_grad_fm_bar}')

ax.axhline(norm_grad_fm_bar, linestyle=':', color=pF[0].get_color(), label=f'$\\|\\nabla\\bar{{F}}\\|_2$ ({norm_grad_fm_bar:.2e})')
ax.axhline(norm_grad_Rm_bar, linestyle=':', color=pR[0].get_color(), label=f'$\\|\\nabla\\bar{{R}}\\|_2$ ({norm_grad_Rm_bar:.3e})')
ax.axhline(norm_grad_Vm_bar, linestyle=':', color=pV[0].get_color(), label=f'$\\|\\nabla\\bar{{V}}\\|_2$ ({norm_grad_Vm_bar:.3e})')

ax.set_xlabel('Iterations')
ax.set_ylabel(r'$\|\nabla_{(w,\theta)}F_m\|_2$')
ax.set_title(r'Evolution of $\|\nabla_{(w,\theta)}F_m\|_2$')
ax.legend()

# Draw some input samples
mean, cov = np.zeros(tln_env.d), 1e4*np.eye(tln_env.d)
x = np.random.multivariate_normal(mean, cov, size=100)

# Plot label of initial network
ax = fig.add_subplot(234)
mean, cov = np.zeros(tln_env.d), np.eye(tln_env.d)
y0 = tln_env.y(w0, theta0, x)
cmap = cm.get_cmap('viridis', 256)
sm = cm.ScalarMappable(cmap=cmap)
colors = sm.to_rgba(y0)
# ax.scatter(x, np.zeros_like(x), label='Sampled x', marker='.', color=colors)
ax.scatter(x[:, 0], x[:, 1], label='Sampled x', marker='.', color=colors)
ax.set_title('Labels of network before SGD')
ax.set_xlabel('$x$')
ax.get_yaxis().set_visible(False)
ax.legend()
plt.colorbar(sm, ax=ax)

# Plot label of converged network
ax = fig.add_subplot(235)
mean, cov = np.zeros(tln_env.d), np.eye(tln_env.d)
y_hat = tln_env.y(w_final, theta_final, x)
cmap = cm.get_cmap('viridis', 256)
sm = cm.ScalarMappable(cmap=cmap)
colors = sm.to_rgba(y_hat)
# ax.scatter(x, np.zeros_like(x), label='Sampled x', marker='.', color=colors)
ax.scatter(x[:, 0], x[:, 1], label='Sampled x', marker='.', color=colors)
ax.set_title('Labels of network after SGD')
ax.set_xlabel('$x$')
ax.get_yaxis().set_visible(False)
ax.legend()
plt.colorbar(sm, ax=ax)

# Plot labels of ground truth
ax = fig.add_subplot(236)
y_bar = tln_env.y_bar(x)
cmap = cm.get_cmap('viridis', 256)
sm = cm.ScalarMappable(cmap=cmap)
colors = sm.to_rgba(y_bar)
# ax.scatter(x, np.zeros_like(x), label='Sampled x', marker='.', color=colors)
ax.scatter(x[:, 0], x[:, 1], label='Sampled x', marker='.', color=colors)
ax.set_title('Ground truth labels')
ax.set_xlabel('$x$')
ax.get_yaxis().set_visible(False)
ax.legend()
plt.colorbar(sm, ax=ax)

plt.tight_layout()
os.makedirs('fig/', exist_ok=True)
plt.savefig(f'fig/tln-b_{tln_env.beta}-{time.time()}.pdf')
plt.show()
