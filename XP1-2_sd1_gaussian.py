"""Implement the experiment 1.2 (gaussian kernel)."""
import numpy as np
import matplotlib.pyplot as plt

import parameters
import sparse_deconvolution_1D as sd1
import optimizer as opt
import plot


np.random.seed(0)

sigma = 0.1
for m in [6, 10, 100]:
    print(f'----------m={m}----------')
    params = parameters.XP12Params(m=m, sigma=sigma, fb_gamma=0.2, lbd=1e-10)
    SD1 = sd1.SparseDeconvolution(params)

    # Apply the forward backward algorithm
    ws, thetas = opt.forward_backward(SD1, print_every=100)

    # Dump arrays
    plot.dump(ws, thetas, params)

    # Plot particle flow
    plot.plot_particle_flow_sd1(ws, thetas, params)

m = 100
for sigma in [1e-3, 1e-1, 1]:
    print(f'----------m={m}----------')
    params = parameters.XP12Params(m=m, sigma=sigma, fb_gamma=0.2, lbd=1e-10)
    SD1 = sd1.SparseDeconvolution(params)

    # Apply the forward backward algorithm
    ws, thetas = opt.forward_backward(SD1, print_every=100)

    # Dump arrays
    plot.dump(ws, thetas, params)

    # Plot particle flow
    plot.plot_particle_flow_sd1(ws, thetas, params)

plt.tight_layout()
plt.show()
