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
    params = parameters.XP11Params(m=m, order=order)
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
                                norm_gradient=norm_gradient,
                                display_legend=(i == 2),
                                )

m = 100
for order in [1, 5, 25]:
    print(f'----------m={order}----------')
    params = parameters.XP11Params(m=m, order=order)
    SD1 = sd1.SparseDeconvolution(params)

    # Apply the forward backward algorithm
    ws, thetas = opt.forward_backward(SD1, print_every=100)

    # Dump arrays
    plot.dump(ws, thetas, params)

    # Plot particle flow
    plot.plot_particle_flow_sd1(ws, thetas, params)

plt.tight_layout()
plt.show()
