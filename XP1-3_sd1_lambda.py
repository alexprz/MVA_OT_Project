"""Implement the experiment 1.3 (influence of lambda)."""
import numpy as np
import matplotlib.pyplot as plt

import parameters
import sparse_deconvolution_1D as sd1
import optimizer as opt
import plot


np.random.seed(0)

m = 100
for lbd in [1, 1e-1, 1e-4]:
    print(f'----------lambda={lbd}----------')
    params = parameters.XP13Params(m=m, lbd=lbd)
    SD1 = sd1.SparseDeconvolution(params)

    # Apply the forward backward algorithm
    ws, thetas = opt.forward_backward(SD1, print_every=100)

    # Dump arrays
    plot.dump(ws, thetas, params)

    # Plot particle flow
    plot.plot_particle_flow_sd1(ws, thetas, params)

plt.tight_layout()
plt.show()
