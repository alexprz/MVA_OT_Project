"""Define a common environment structure."""
from collections import namedtuple


Env = namedtuple('Env', [
    'R',
    'phi',
    'V',
    'y',
    'g',
    'w',
    'p',
    'x_min',
    'x_max',
    'grad_R',
    'psi',
    'psi_p',
    'prox_V',
    'subgrad_V',
    'lbd',
])

NNEnv = namedtuple('NNEnv', [
    'd',
    'phi',
    'phi_dw',
    'phi_dtheta',
    'V',
    'V_dw',
    'V_dtheta',
    'y',
    'w_bar',
    'theta_bar',
    'loss',
    'loss_d1',
    'forward',
    'sigma',
    'sigma_d',
    'Rm',
    'Vm',
    'grad_Rm',
    'subgrad_Vm',
])
