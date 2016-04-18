import numpy as np

from diffraction import Site


# Crystal Structure
lattice_parameters = (5.5876, 5.5876, 13.867, 90, 90, 120)
chemical_formula = 'BiFeO3'
formula_units = 6
g_factor = 2
space_group = 'R3c'

# Atomic sites
sites = {
    'Bi1': Site('Bi3+', (0,  0,  0)),
    'Fe1': Site('Fe3+', (0,  0,  0.2212)),
    'O1': Site('O2-', (0.443,  0.012,  0.9543))
}

# Magnetic sites
magnetic_ion = sites['Fe1'].ion
pos_Fe1 = sites['Fe1'].position
pos_Fe2 = pos_Fe1

# Propagation vectors
delta = 0.0045
k1 = np.array((delta, delta, 0))
k2 = np.array((delta, -2 * delta, 0))
k3 = np.array((-2 * delta, delta, 0))
