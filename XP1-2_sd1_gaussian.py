"""Implement the experiment 1.2 (gaussian kernel)."""
import numpy as np
import matplotlib.pyplot as plt

import parameters
import sparse_deconvolution_1D as sd1
import optimizer as opt
import plot


np.random.seed(0)
w_compare, theta_compare = None, None
sigma = 0.01
for i, m in enumerate([100, 10, 6]):
    print(f'----------m={m}----------')
    if m == 100:
        params = parameters.XP12Params(m=m, sigma=sigma, lbd=2, fb_gamma=0.005, fb_lbd=0.01)
    else:
        params = parameters.XP12Params(m=m, sigma=sigma, lbd=2, fb_gamma=0.0001, fb_lbd=0.01)
    SD1 = sd1.SparseDeconvolution(params)

    # Apply the forward backward algorithm
    ws, thetas = opt.forward_backward(SD1, print_every=100)

    # The first result in the one of reference
    if w_compare is None or theta_compare is None:
        w_compare, theta_compare = ws[-1, ...], thetas[-1, ...]
        val_compare = m

    # Dump arrays
    plot.dump(ws, thetas, params)

    # Plot particle flow
    grad_w, subgrad_theta = SD1.subgrad_f_m(ws[-1, :], thetas[-1, :])
    norm_gradient = np.linalg.norm(grad_w) + np.linalg.norm(subgrad_theta)
    plot.plot_particle_flow_sd1(ws, thetas, params, w_compare=None,
                                theta_compare=None,
                                tol_compare=1e-1,
                                label_compare=f'Converged positions\nfor m={val_compare}',
                                norm_gradient=None,  # norm_gradient,
                                display_legend=(i == 2),
                                )

np.random.seed(0)
w_compare, theta_compare = None, None
m = 100
for i, sigma in enumerate([5e-3, 1e-1, 1]):
    print(f'----------m={m}----------')
    if i == 0:
        params = parameters.XP12Params(m=m, sigma=sigma, n_iter=10000, lbd=2, fb_gamma=0.003, fb_lbd=0.01)
    elif i == 1:
        params = parameters.XP12Params(m=m, sigma=sigma, n_iter=10000, lbd=6, fb_gamma=0.003, fb_lbd=0.01)
    else:
        params = parameters.XP12Params(m=m, sigma=sigma, n_iter=10000, lbd=0.5, fb_gamma=0.003, fb_lbd=0.01)
    SD1 = sd1.SparseDeconvolution(params)

    # Apply the forward backward algorithm
    ws, thetas = opt.forward_backward(SD1, print_every=100)

    # The first result in the one of reference
    if w_compare is None or theta_compare is None:
        w_compare, theta_compare = ws[-1, ...], thetas[-1, ...]
        val_compare = sigma

    # Dump arrays
    plot.dump(ws, thetas, params)

    # Plot particle flow
    plot.plot_particle_flow_sd1(ws, thetas, params, w_compare=None,
                                theta_compare=None,
                                tol_compare=1e-1,
                                label_compare=f'Converged positions\nfor sigma={val_compare}',
                                norm_gradient=None,  # norm_gradient,
                                display_legend=(i == 2),
                                )

plt.show()
