"""Implement the experiment 2.4 (ReLU + logistic loss)."""
import numpy as np
import matplotlib.pyplot as plt

import parameters
import two_layer_nn as tln
import optimizer as opt
import losses
import plot


np.random.seed(0)
w_compare, theta_compare = None, None

# m = 100
# params_compare = parameters.XP21Params(m=m)
# TLN = tln.TwoLayerNN(params_compare)
# ws, thetas, *_ = opt.SGD(TLN, print_every=100)
# w_compare, theta_compare = ws[-1, ...], thetas[-1, ...]
# val_compare = params_compare.lbd
# val_compare = m

for i, m in enumerate([100, 10, 6]):
    print(f'----------m={m}----------')
    params = parameters.XP24Params(m=m, loss=losses.Logistic())
    TLN = tln.TwoLayerNN(params)

    # Apply the forward backward algorithm
    ws, thetas, *_ = opt.SGD(TLN, print_every=100)

    # The first result in the one of reference
    if w_compare is None or theta_compare is None:
        w_compare, theta_compare = ws[-1, ...], thetas[-1, ...]
        val_compare = m

    # Dump arrays
    plot.dump(ws, thetas, params)

    # Plot particle flow
    plot.plot_particle_flow_tln(ws, thetas, params, w_compare, theta_compare,
                                tol_compare=1e-1,
                                label_compare=f'Optimal positions\nfor $m={val_compare}$',
                                display_legend=(i == 2),
                                )

plt.show()
