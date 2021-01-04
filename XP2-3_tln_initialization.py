"""Implement the experiment 2.3 (influence of initialization)."""
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
log_roots = np.exp(-2*np.pi*1j*np.logspace(-10, 0, m))
log_circle = np.stack((np.real(log_roots), np.imag(log_roots)), axis=1)

neg_roots = np.exp(-np.pi*1j*np.linspace(0, 1, m))
neg_circle = np.stack((np.real(neg_roots), np.imag(neg_roots)), axis=1)

theta_line = np.stack((np.linspace(0, 1, m), np.zeros(m)), axis=1)

initializations = {
    # 'no_separability': (5e-1*np.ones(m), np.zeros((m, 2))),
    'overspread': (1e1*np.ones(m), uniform_circle),
    'unbalanced_sep': (1e-1*np.ones(m), log_circle),
    'unbalanced_dist': (1e-1*np.ones(m), theta_line),
    'half_space': (np.ones(m), neg_circle),
}

params_compare = parameters.XP21Params(m=m)
TLN = tln.TwoLayerNN(params_compare)
ws, thetas, *_ = opt.SGD(TLN, print_every=100)
w_compare, theta_compare = ws[-1, ...], thetas[-1, ...]
val_compare = params_compare.lbd
val_compare = m

# w_compare, theta_compare = None, None

for i, (name, (w0, theta0)) in enumerate(initializations.items()):
    print(f'----------{name}----------')
    if i == 0:
        params = parameters.XP23Params(w0=w0, theta0=theta0, name=name,
                                       sgd_gamma=1e-2)
    else:
        params = parameters.XP23Params(w0=w0, theta0=theta0, name=name,
                                       sgd_gamma=1)

    TLN = tln.TwoLayerNN(params)

    # Apply the forward backward algorithm
    ws, thetas, *_ = opt.SGD(TLN, print_every=100)

    # The first result in the one of reference
    if w_compare is None or theta_compare is None:
        w_compare, theta_compare = ws[-1, ...], thetas[-1, ...]
        val_compare = name

    # Dump arrays
    plot.dump(ws, thetas, params)

    # Plot particle flow
    plot.plot_particle_flow_tln(ws, thetas, params,
                                w_compare,
                                theta_compare,
                                tol_compare=1e-1,
                                label_compare=f'Optimal positions\nfor $m={val_compare}$',
                                display_legend=(i == 1),
                                )

plt.tight_layout()
plt.show()
