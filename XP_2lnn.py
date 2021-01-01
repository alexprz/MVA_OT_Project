"""Implement the one hidden layer experiment of the paper."""
import os
import time
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

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
    ax.plot(100*x, 100*y, **kwargs)


np.random.seed(0)
m0 = 4
tln_env = tln.paper_env(m0, act.ReLU(), losses.Squared(), beta=1e-1)

# Initialize the particle flow
m = 10
eps = 1e-1
w0 = eps*np.ones(m)
roots = np.array([np.exp(2*np.pi*1j*k/m) for k in range(m)])[:, None]
theta0 = np.concatenate((np.real(roots), np.imag(roots)), axis=1)
w_bar = np.zeros_like(w0)
theta_bar = np.zeros_like(theta0)
w_bar[:m0] = m/m0*tln_env.w_bar
theta_bar[:m0, :] = tln_env.theta_bar
w0 = np.copy(w_bar)
theta0 = np.copy(theta_bar)


bs = 100
n_iter = 1000
gamma0 = 1e-1
ws, thetas, fms, norm_grad_fms, Rms, Vms, norm_grad_Rms, norm_grad_Vms = opt.SGD(tln_env, w0, theta0, bs, n_iter, gamma0, print_every=10)

w_final, theta_final = ws[-1, ...], thetas[-1, ...]


# Plot objective evolution
mean, cov = np.zeros(tln_env.d), np.eye(tln_env.d)
x = np.random.multivariate_normal(mean, cov, size=1000)
fm_bar = tln_env.fm(w_bar, theta_bar, x)
Rm_bar = tln_env.Rm(w_bar, theta_bar, x)
Vm_bar = tln_env.Vm(w_bar, theta_bar)

fig = plt.figure(figsize=(18, 6))
ax = fig.add_subplot(132)
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

ax = fig.add_subplot(133)
pF = ax.plot(norm_grad_fms, label='$\\|\\nabla F_m\\|_2 = \\|\\nabla R_m + \\nabla V_m\\|_2$')
pR = ax.plot(norm_grad_Rms, label='$\\|\\nabla R_m\\|_2$', linewidth=.5)
pV = ax.plot(norm_grad_Vms, label='$\\|\\nabla V_m\\|_2$', linewidth=.5)
print(f'Gradient optimal fm: {norm_grad_fm_bar}')

ax.axhline(norm_grad_Rm_bar, linestyle=':', color=pR[0].get_color(), label=f'$\\|\\nabla\\bar{{R}}\\|_2$ ({norm_grad_Rm_bar:.3e})')
ax.axhline(norm_grad_Vm_bar, linestyle=':', color=pV[0].get_color(), label=f'$\\|\\nabla\\bar{{V}}\\|_2$ ({norm_grad_Vm_bar:.3e})')
ax.axhline(norm_grad_fm_bar, linestyle=':', color=pF[0].get_color(), label=f'$\\|\\nabla\\bar{{F}}\\|_2$ ({norm_grad_fm_bar:.2e})')

ax.set_xlabel('Iterations')
ax.set_ylabel(r'$\|\nabla_{(w,\theta)}F_m\|_2$')
ax.set_title(r'Evolution of $\|\nabla_{(w,\theta)}F_m\|_2$')
ax.legend()

# Plot ground truth
ax2 = fig.add_subplot(131)

scatterplot(w0, theta0, ax2, color='blue', marker='.')
scatterplot(w_final, theta_final, ax2, color='red', marker='.')
# ax2.scatter(w0*theta0[:, 0], w0*theta0[:, 1], color='blue', marker='.')
# ax2.scatter(w_final*theta_final[:, 0], w_final*theta_final[:, 1], color='red', marker='.')
ax2.set_xlabel(r'$\theta_1$')
ax2.set_ylabel(r'$\theta_2$')
ax2.set_title('Trajectories of the particles')

# x_min, x_max = ax2.get_xlim()
# y_min, y_max = ax2.get_ylim()
# print(x_min, x_max)
# print(y_min, y_max)
w_bar = tln_env.w_bar
theta_bar = tln_env.theta_bar
# xx = np.linspace(0, 2, 2)
# XX = np.sign(theta_bar[:, 0])[:, None]*xx[None, :]
# YY = np.divide(theta_bar[:, 1], theta_bar[:, 0])[:, None]*xx[None, :]
# XX = np.clip(XX, -1, 1)
# YY = np.clip(YY, -1, 1)
# for k in range(m0):
#     # plt.plot(np.sign(theta_bar[k, 0])*XX, theta_bar[k, 1]/theta_bar[k, 0]*XX, linestyle='--', color='black')
#     label = 'Optimal positions' if k == 0 else ''
#     ax2.plot(XX[k, :], YY[k, :], linestyle='--', color='black', label=label)
#     # ax2.scatter(w_bar[k]*XX[k, -1], w_bar[k]*YY[k, -1], marker='+', color='orange')

scatterplot(w_bar, theta_bar, ax2, marker='+', color='orange')

# Plot paths
for k in range(m):
    label = 'Flow' if k == 0 else ''
    ax2.plot(ws[:, k]*thetas[:, k, 0], ws[:, k]*thetas[:, k, 1], color='green', linewidth=1, label=label)#, marker='o', markersize=1)

# max_ylim = max(abs(y_min), abs(y_max))
# ax = plt.gca()
# ax.set_ylim(-2, 2)
x_min, x_max = ax2.get_xlim()
y_min, y_max = ax2.get_ylim()
lineplot(w_bar, theta_bar, ax2, linestyle='--', color='black', label='Optimal positions')
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)
ax2.legend()

plt.tight_layout()

os.makedirs('fig/', exist_ok=True)
plt.savefig(f'fig/tln-b_{tln_env.beta}-{time.time()}.pdf')

plt.show()
