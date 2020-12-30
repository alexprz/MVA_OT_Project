"""Implement the sparse deconvolution experience of the paper."""
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

m0 = 5
env = sd1.paper_env(m0)
# print(env)

# plt.scatter(env.p, env.w, color='red', marker='+')
ax = plt.gca()
for i, p in enumerate(env.p):
    sign = np.sign(env.w[i])
    ymin, ymax = (0.5, 1) if sign > 0 else (0, 0.5)
    print(ymin, ymax)
    label = 'Optimal positions' if i == 0 else ''
    ax.axvline(p, ymin=ymin, ymax=ymax, color='black', linestyle='--',
               linewidth=1, label=label)


n = 1000
XX = np.linspace(0, 1, n)
# YY = env.y(XX)
# plt.plot(XX, YY, label='y true')

# r = env.R(np.ones((3, 100)))
# print('r', r)


# plt.plot(XX, dirichlet_kernel(2*np.pi*XX, n=7))
# plt.plot(XX, dirichlet_kernel_dx(2*np.pi*XX, n=7))
# plt.show()
# exit()
# plt.plot(XX, env.psi(XX))
# plt.plot(XX, env.psi_p(XX))



# plt.show()
# exit()

# w0 = np.arange(10)/10
# theta0 = np.arange(10)/10
# w0 = env.w
m = 30
w0 = np.zeros(m)
theta0 = np.arange(m)/m
n = 100
max_iter = 300000

plt.scatter(theta0, w0, color='blue', marker='.')

ws, thetas = opt.forward_backward(env, w0, theta0, max_iter=max_iter, n=n)

w_final = ws[-1, :]
theta_final = thetas[-1, :]

plt.scatter(theta_final, w_final, color='red', marker='.', label='Particle')
for k in range(ws.shape[1]):
    label = 'Flow' if k == 0 else ''
    plt.plot(thetas[:, k], ws[:, k], color='green', linewidth=0.8, label=label)

# y_approx = np.dot(env.psi(XX[:, None] - thetas[-1, None, :]), w_final)
#y_approx2 = np.dot(env.psi(XX[:, None] - env.p[None, :]), env.w)
#y_approx = sd1.y(XX, w_final, theta_final, sd1.psi)
# plt.plot(XX, y_approx, label='y approx')
# plt.plot(XX, y_approx2, label='y approx 2')
# for i in range(0, 100, 10):
#     y_approx = np.dot(env.psi(XX[:, None] - thetas[i, None, :]), w_final)
#     plt.plot(XX, y_approx, label='y approx')
plt.legend()




y_min, y_max = ax.get_ylim()
max_ylim = max(abs(y_min), abs(y_max))
ax.set_ylim(-max_ylim, max_ylim)
plt.show()

exit()

# w = np.arange(10)
# theta = np.arange(10)
# grad_w, grad_theta = env.grad_R(w, theta, XX)
# print(grad_w.shape, grad_theta.shape)



phi = env.phi(w, theta, XX)
for i in range(len(w)):
    plt.plot(XX, phi[i, :])

w = np.ones(10)
theta = np.arange(10)
val = f_m(w, theta, env, n=1000)
print('val', val)
# plt.show()
