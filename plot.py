"""Implement functions to plot results."""
import os
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    'text.usetex': True,
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
    'font.size': 15,
    'figure.figsize': (12, 6),
})


def get_ax(ax):
    """Create an ax if ax is None."""
    if ax is None:
        fig = plt.figure()
        return fig, plt.gca()
    return None, ax


def get_XP_folder(params):
    folder = f'fig/{params.name}/'
    os.makedirs(folder, exist_ok=True)
    return folder


def dump(ws, thetas, params):
    """Dump optimization results in a dedicated folder."""
    folder = get_XP_folder(params)
    ws.dump(f'{folder}ws{params}.pickle')
    thetas.dump(f'{folder}thetas{params}.pickle')


def savefig(params, name=''):
    """Save current fig in pdf and jpg."""
    folder = get_XP_folder(params)
    plt.savefig(f'{folder}{name}{params}.pdf')
    plt.savefig(f'{folder}{name}{params}.jpg')


def plot_particle_flow_sd1(ws, thetas, params, ax=None):
    """Plot the particle flow in sparse deconvolution.

    Args:
    -----
        ws : np.array of shape (n_iter, m)
        thetas : np.array of shape (n_iter, m)
        params : parameters.BaseParameters object
        ax : matplotlib ax

    """
    fig, ax = get_ax(ax)

    # Print the optimal positions
    for i, theta in enumerate(params.theta_bar):
        sign = np.sign(params.w_bar[i])
        ymin, ymax = (0.5, 1) if sign > 0 else (0, 0.5)
        label = 'Optimal positions' if i == 0 else ''
        ax.axvline(theta, ymin=ymin, ymax=ymax, color='black', linestyle='--',
                   linewidth=1, label=label)

    # Plot initial particles
    ax.scatter(params.theta0, params.w0, color='blue', marker='.')

    # Plot the final particles
    ax.scatter(thetas[-1, :], ws[-1, :], color='red', marker='.', label='Particle')

    # Plot the particles' trajectories during optimization
    for k in range(params.m):
        label = 'Flow' if k == 0 else ''
        plt.plot(thetas[:, k], ws[:, k], color='green', linewidth=0.8, label=label)

    # Center the plot on the y axis
    y_min, y_max = ax.get_ylim()
    max_ylim = max(abs(y_min), abs(y_max))
    ax.set_ylim(-max_ylim, max_ylim)
    ax.legend()

    if fig is not None:
        plt.tight_layout()
        savefig(params, name='particle_flow')

    return ax
