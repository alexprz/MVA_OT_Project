"""Implement the sparse deconvolution experience of the paper."""
import numpy as np
import matplotlib.pyplot as plt

import parameters
import sparse_deconvolution_1D as sd1
import optimizer as opt
import plot


np.random.seed(0)

m = 100
# We study 3 different initialisations
initializations = {
    'no_separability': (np.zeros(m), .5*np.ones(m)),
    'unbalanced_dist': (np.zeros(m), np.arange(m)/(10*m)),
    'unbalanced_sep': (np.zeros(m), np.logspace(-10, 0, m)),
    'non_zero_weights': (np.ones(m), np.arange(m)/m),
}

for name, (w0, theta0) in initializations.items():
    print(f'----------{name}----------')
    params = parameters.XP14Params(w0=w0, theta0=theta0, name=name)
    SD1 = sd1.SparseDeconvolution(params)

    # Apply the forward backward algorithm
    ws, thetas = opt.forward_backward(SD1, print_every=100)

    # Dump arrays
    plot.dump(ws, thetas, params)

    # Plot particle flow
    plot.plot_particle_flow_sd1(ws, thetas, params)

plt.tight_layout()
plt.show()
