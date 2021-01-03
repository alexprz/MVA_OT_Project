"""Implement the sparse deconvolution experience of the paper."""
import numpy as np
import matplotlib.pyplot as plt

import parameters
import two_layer_nn as tln
import optimizer as opt
import losses
import plot


np.random.seed(0)

for m in [6, 10, 100]:
    print(f'----------m={m}----------')
    params = parameters.XP24Params(m=m, loss=losses.Logistic())
    TLN = tln.TwoLayerNN(params)

    # Apply the forward backward algorithm
    ws, thetas, *_ = opt.SGD(TLN, print_every=100)

    # Dump arrays
    plot.dump(ws, thetas, params)

    # Plot particle flow
    plot.plot_particle_flow_tln(ws, thetas, params)

plt.tight_layout()
plt.show()
