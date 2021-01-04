"""Implement the experiment 2.2 (influence of lambda)."""
import numpy as np
import matplotlib.pyplot as plt

import parameters
import two_layer_nn as tln
import optimizer as opt
import plot


np.random.seed(0)

m = 100
w_compare, theta_compare = None, None

for i, lbd in enumerate([1e-3, 1.2e-1, 1e-5]):
    print(f'----------m={m}----------')
    params = parameters.XP22Params(m=m, lbd=lbd, sgd_n_iter=10000)
    TLN = tln.TwoLayerNN(params)

    # Apply the forward backward algorithm
    ws, thetas, *_ = opt.SGD(TLN, print_every=100)

    # The first result in the one of reference
    if w_compare is None or theta_compare is None:
        w_compare, theta_compare = ws[-1, ...], thetas[-1, ...]
        val_compare = lbd

    # Dump arrays
    plot.dump(ws, thetas, params)

    # Plot particle flow
    plot.plot_particle_flow_tln(ws, thetas, params, w_compare, theta_compare,
                                tol_compare=1e-1,
                                label_compare=f'Optimal positions\nfor $\\lambda={val_compare}$',
                                display_legend=(i == 2),
                                )

plt.show()
