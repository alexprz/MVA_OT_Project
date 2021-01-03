"""Implement the experiment 2.2 (influence of lambda)."""
import numpy as np
import matplotlib.pyplot as plt

import parameters
import two_layer_nn as tln
import optimizer as opt
import plot


np.random.seed(0)

m = 100
for lbd in [1e6, 1e3, 1e1]:
    print(f'----------m={m}----------')
    params = parameters.XP22Params(m=m, lbd=lbd)
    TLN = tln.TwoLayerNN(params)

    # Apply the forward backward algorithm
    ws, thetas, *_ = opt.SGD(TLN, print_every=100)

    # Dump arrays
    plot.dump(ws, thetas, params)

    # Plot particle flow
    plot.plot_particle_flow_tln(ws, thetas, params)

plt.tight_layout()
plt.show()
