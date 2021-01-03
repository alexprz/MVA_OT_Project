"""Implement the sparse deconvolution experience of the paper."""
import numpy as np
import matplotlib.pyplot as plt

import parameters
import two_layer_nn as tln
import optimizer as opt
import plot


np.random.seed(0)

m = 100
roots = np.array([np.exp(2*np.pi*1j*k/m) for k in range(m)])[:, None]
uniform_circle = np.concatenate((np.real(roots), np.imag(roots)), axis=1)
log_roots = np.exp(2*np.pi*1j*np.logspace(-10, 0, m))
log_circle = np.stack((np.real(log_roots), np.imag(log_roots)), axis=1)

theta_line = np.stack((np.arange(m), np.zeros(m)), axis=1)

initializations = {
    'no_separability': (np.zeros(m), np.zeros((m, 2))),
    'overspread': (1e1*np.ones(m), uniform_circle),
    'unbalanced_sep': (1e-1*np.ones(m), log_circle),
    'unbalanced_dist': (np.ones(m), theta_line),
}

for name, (w0, theta0) in initializations.items():
    print(f'----------{name}----------')
    params = parameters.XP23Params(w0=w0, theta0=theta0, name=name, sgd_gamma=1e-2)
    TLN = tln.TwoLayerNN(params)

    # Apply the forward backward algorithm
    ws, thetas, *_ = opt.SGD(TLN, print_every=100)

    # Dump arrays
    plot.dump(ws, thetas, params)

    # Plot particle flow
    plot.plot_particle_flow_tln(ws, thetas, params)

plt.tight_layout()
plt.show()
