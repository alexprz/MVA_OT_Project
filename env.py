"""Define a common environment structure."""
from collections import namedtuple


Env = namedtuple('Env', '''R phi V y g w p x_min, x_max grad_R psi psi_p
prox_V subgrad_V''')
