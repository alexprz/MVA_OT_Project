"""Implement the sparse deconvolution experience of the paper."""
import numpy as np
import matplotlib.pyplot as plt

import parameters
import sparse_deconvolution_1D as sd1
import optimizer as opt
import plot


np.random.seed(0)

order = 7
for m in [6, 10, 100]:
    print(f'----------m={m}----------')
    params = parameters.XP11Params(m=m, order=order)
    SD1 = sd1.SparseDeconvolution(params)

    # Apply the forward backward algorithm
    ws, thetas = opt.forward_backward(SD1, print_every=100)

    # Dump arrays
    plot.dump(ws, thetas, params)

    # Plot particle flow
    plot.plot_particle_flow_sd1(ws, thetas, params)

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
