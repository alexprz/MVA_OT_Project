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


def get_ax(ax, figsize=None):
    """Create an ax if ax is None."""
    if ax is None:
        fig = plt.figure(figsize=figsize)
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


def plot_particle_flow_sd1(ws, thetas, params, w_compare=None,
                           theta_compare=None, tol_compare=None,
                           label_compare=None, norm_gradient=None,
                           display_legend=True, figsize=(9, 6), ax=None):
    """Plot the particle flow in sparse deconvolution.

    Args:
    -----
        ws : np.array of shape (n_iter, m)
        thetas : np.array of shape (n_iter, m)
        params : parameters.BaseParameters object
        w_compare : np.array of shape (m,)
        theta_compare : np.array of shape (m,)
        ax : matplotlib ax

    """
    fig, ax = get_ax(ax, figsize)

    # Print 0 line
    ax.axhline(0, color='black', linestyle='-', linewidth=1)

    # Print the positions of ground truth
    for i, theta in enumerate(params.theta_bar):
        sign = np.sign(params.w_bar[i])
        ymin, ymax = (0.5, 1) if sign > 0 else (0, 0.5)
        label = 'Truth positions' if i == 0 else None
        ax.axvline(theta, ymin=ymin, ymax=ymax, color='black', linestyle='--',
                   linewidth=1, label=label)

    # Print the positions to compare
    label = None
    if w_compare is not None and theta_compare is not None:
        for i, w in enumerate(w_compare):
            if tol_compare is not None and abs(w) < tol_compare:
                continue
            sign = np.sign(w)
            ymin, ymax = (0.5, 1) if sign > 0 else (0, 0.5)
            label = label_compare if label is None else ''
            ax.axvline(theta_compare[i], ymin=ymin, ymax=ymax, color='cyan',
                       linestyle='--', linewidth=1, label=label)

    # Plot initial particles
    ax.scatter(params.theta0, params.w0, color='blue', marker='.', label='Initial particle\npositions')

    # Plot the final particles
    label = 'Final particles'
    if norm_gradient is not None:
        label = f'{label}\n$\\|\\partial F_m\\|_2={norm_gradient:.0e}$'
    ax.scatter(thetas[-1, :], ws[-1, :], color='red', marker='.', label=label)

    # Plot the particles' trajectories during optimization
    for k in range(params.m):
        label = 'Flow' if k == 0 else ''
        plt.plot(thetas[:, k], ws[:, k], color='green', linewidth=0.8, label=label)

    # Center the plot on the y axis
    y_min, y_max = ax.get_ylim()
    max_ylim = max(abs(y_min), abs(y_max))
    ax.set_ylim(-max_ylim, max_ylim)
    if display_legend:
        ax.legend()

    # ax.set_xlabel('$\\theta$')
    # ax.set_ylabel('$w$')

    if fig is not None:
        plt.tight_layout(pad=0.1)
        savefig(params, name='particle_flow')

    return ax


def scatterplot(w, theta, ax, **kwargs):
    """Scatter particles."""
    x = w*theta[:, 0]
    y = w*theta[:, 1]
    ax.scatter(x, y, **kwargs)


def lineplot(w, theta, ax, **kwargs):
    """Draw lines between (0, 0) and particles."""
    w = np.array(w).reshape(-1)
    theta = np.array(theta)
    if theta.ndim == 1:
        theta = theta.reshape(1, -1)
    x = np.stack((np.zeros_like(w), w*theta[:, 0]), axis=0)
    y = np.stack((np.zeros_like(w), w*theta[:, 1]), axis=0)
    ax.plot(1e4*x, 1e4*y, **kwargs)


def plot_particle_flow_tln(ws, thetas, params, w_compare=None,
                           theta_compare=None, tol_compare=None,
                           label_compare=None,
                           display_legend=True, figsize=(9, 6), ax=None):
    """Plot the particle flow in two-layers network example.

    Args:
    -----
        ws : np.array of shape (n_iter, m)
        thetas : np.array of shape (n_iter, m)
        params : parameters.BaseParameters object
        ax : matplotlib ax

    """
    w_final, theta_final = ws[-1, ...], thetas[-1, ...]
    fig, ax = get_ax(ax, figsize)

    # Plot particle paths and start/end
    scatterplot(params.w0, params.theta0, ax, color='blue', marker='.', label='Initial particles', zorder=3)
    for k in range(params.m):
        label = 'Flow' if k == 0 else ''
        ax.plot(ws[:, k]*thetas[:, k, 0], ws[:, k]*thetas[:, k, 1], color='lightgreen', linewidth=.1, label=label, zorder=4)#, marker='o', markersize=1)
    scatterplot(w_final, theta_final, ax, color='red', marker='.', s=50, label='Final particles', zorder=5)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Plot ground truth
    # scatterplot(params.w_bar, params.theta_bar, ax, marker='+', color='orange', label='Truth positions', zorder=1)

    # Plot lines of truth positions
    label = None
    for i, w in enumerate(params.w_bar):
        label = 'Truth directions' if label is None else ''
        lineplot(w, params.theta_bar[i, :], ax, linestyle='--', linewidth=0.5, color='black', label=label, zorder=0)

    # Plot lines of positions to compare
    label = None
    if w_compare is not None and theta_compare is not None:
        for i, w in enumerate(w_compare):
            if tol_compare is not None and abs(w) < tol_compare:
                continue
            label = label_compare if label is None else ''
            lineplot(w, theta_compare[i, :], ax, linestyle='-', linewidth=0.8, color='c', label=label, zorder=2)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    if display_legend:
        ax.legend()

    if fig is not None:
        plt.tight_layout(pad=0.1)
        savefig(params, name='particle_flow')

    return ax
