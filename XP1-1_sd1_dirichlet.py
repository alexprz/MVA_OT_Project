"""Implement the experiment 1.1 (dirichlet kernel)."""
import numpy as np
import matplotlib.pyplot as plt

import parameters
import sparse_deconvolution_1D as sd1
import optimizer as opt
import plot


np.random.seed(0)
w_compare, theta_compare = None, None
order = 7
for i, m in enumerate([100, 10, 6]):
    print(f'----------m={m}----------')
    params = parameters.XP11Params(m=m, order=order, fb_gamma=0.03, fb_lbd=0.1)
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
    plot.plot_particle_flow_sd1(ws, thetas, params, w_compare, theta_compare,
                                tol_compare=1e-1,
                                label_compare=f'Converged positions\nfor m={val_compare}',
                                norm_gradient=None,  # norm_gradient,
                                display_legend=(i == 2),
                                )

np.random.seed(0)
w_compare, theta_compare = None, None
m = 100
for i, order in enumerate([25, 5, 1]):
    print(f'----------m={order}----------')
    if order == 25:
        params = parameters.XP11Params(m=m, order=order, n_iter=10000, lbd=0.5, fb_gamma=0.03, fb_lbd=0.1)
    elif order == 5:
        params = parameters.XP11Params(m=m, order=order, n_iter=10000, lbd=0.2, fb_gamma=0.3, fb_lbd=0.1)
    elif order == 1:
        params = parameters.XP11Params(m=m, order=order, n_iter=10000, lbd=0.2, fb_gamma=1, fb_lbd=0.1)
    SD1 = sd1.SparseDeconvolution(params)

    # Apply the forward backward algorithm
    ws, thetas = opt.forward_backward(SD1, print_every=100)

    # The first result in the one of reference
    if w_compare is None or theta_compare is None:
        w_compare, theta_compare = ws[-1, ...], thetas[-1, ...]
        val_compare = order

    # Dump arrays
    plot.dump(ws, thetas, params)

    # Plot particle flow
    grad_w, subgrad_theta = SD1.subgrad_f_m(ws[-1, :], thetas[-1, :])
    norm_gradient = np.linalg.norm(grad_w) + np.linalg.norm(subgrad_theta)
    plot.plot_particle_flow_sd1(ws, thetas, params, w_compare=None, theta_compare=None,
                                tol_compare=1e-1,
                                label_compare=f'Converged positions\nfor n={val_compare}',
                                norm_gradient=None,  # norm_gradient,
                                display_legend=(i == 2),
                                )


plt.show()
