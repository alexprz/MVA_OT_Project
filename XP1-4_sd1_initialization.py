"""Implement the experiment 1.4 (initialization influence)."""
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
# We study 3 different initialisations
initializations = {
    'no_separability': (np.zeros(m), .5*np.ones(m)),
    'unbalanced_dist': (np.zeros(m), np.arange(m)/(10*m)),
    'unbalanced_sep': (np.zeros(m), np.logspace(-10, 0, m)),
    # 'non_zero_weights': (-np.ones(m), np.arange(m)/m),
    'non_zero_weights': (-np.linspace(-1, 1, m), np.arange(m)/m),
}

for i, (name, (w0, theta0)) in enumerate(initializations.items()):
    print(f'----------{name}----------')
    if i == 0 or i == 2:
        params = parameters.XP14Params(w0=w0, theta0=theta0, name=name)
    elif i == 1:
        params = parameters.XP14Params(w0=w0, theta0=theta0, name=name, lbd=0.02)
    elif i == 3:
        # params = parameters.XP14Params(w0=w0, theta0=theta0, name=name, lbd=.5, fb_gamma=1.99)
        # params = parameters.XP14Params(w0=w0, theta0=theta0, name=name, n_iter=30000, lbd=.5, kernel=kernels.DirichletKernel(1, 25), fb_gamma=1.99)
        # params = parameters.XP14Params(w0=w0, theta0=theta0, name=name, n_iter=20000, lbd=.5, kernel=kernels.DirichletKernel(1, 25), fb_gamma=1.99)#, fb_lbd=0.1)
        # params = parameters.XP14Params(w0=w0, theta0=theta0, name=name, n_iter=20000, lbd=.5, fb_gamma=1.99)#, fb_lbd=0.1)
        params = parameters.XP14Params(w0=w0, theta0=theta0, name=name, lbd=.5)#, fb_lbd=0.1)
    SD1 = sd1.SparseDeconvolution(params)

    # Apply the forward backward algorithm
    ws, thetas = opt.forward_backward(SD1, print_every=100)

    # The first result in the one of reference
    if w_compare is None or theta_compare is None:
        w_compare, theta_compare = ws[-1, ...], thetas[-1, ...]
        val_compare = name

    # Dump arrays
    plot.dump(ws, thetas, params)

    # Plot particle flow
    plot.plot_particle_flow_sd1(ws, thetas, params, w_compare=None,
                                theta_compare=None,
                                tol_compare=1e-1,
                                label_compare=f'Converged positions\nfor {val_compare}',
                                norm_gradient=None,  # norm_gradient,
                                display_legend=(i == 2),
                                )

plt.tight_layout()
plt.show()
