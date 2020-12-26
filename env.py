"""Define a common environment structure."""
from collections import namedtuple


Env = namedtuple('Env', 'R phi V y g w p x_min, x_max')
