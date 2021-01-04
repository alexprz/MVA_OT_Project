"""Implement the experiment 1.3 (influence of lambda)."""
import numpy as np
import matplotlib.pyplot as plt

import parameters
import sparse_deconvolution_1D as sd1
import optimizer as opt
import kernels
import plot


np.random.seed(0)
w_compare, theta_compare = None, None
m = 100
# for i, lbd in enumerate([3e-1, 5e-1, 1e-4]):
for i, lbd in enumerate([2e-1, 5e-1, 1e-2]):
    print(f'----------lambda={lbd}----------')
    # params = parameters.XP13Params(m=m, lbd=lbd)
    params = parameters.XP13Params(m=m, lbd=lbd, kernel=kernels.DirichletKernel(1, 25), fb_gamma=0.03, fb_lbd=0.1)
    SD1 = sd1.SparseDeconvolution(params)

    # Apply the forward backward algorithm
    ws, thetas = opt.forward_backward(SD1, print_every=100)

    # The first result in the one of reference
    if w_compare is None or theta_compare is None:
        w_compare, theta_compare = ws[-1, ...], thetas[-1, ...]
        val_compare = lbd

    # Dump arrays
    plot.dump(ws, thetas, params)

    # Plot particle flow
    grad_w, subgrad_theta = SD1.subgrad_f_m(ws[-1, :], thetas[-1, :])
    norm_gradient = np.linalg.norm(grad_w) + np.linalg.norm(subgrad_theta)
    plot.plot_particle_flow_sd1(ws, thetas, params, w_compare=None,
                                theta_compare=None,
                                tol_compare=1e-1,
                                label_compare=f'Converged positions\nfor lbd={val_compare}',
                                norm_gradient=None,  # norm_gradient,
                                display_legend=(i == 2),
                                )

plt.tight_layout()
plt.show()
