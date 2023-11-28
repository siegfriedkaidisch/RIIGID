from ase import Atoms
from ase.calculators.vasp.vasp import Vasp

from RIGID import RIGID

# User defines full system and specifies the indices of a molecule
full_system = Atoms()
molecule_indices = [1,2,3]

# User instantiates a RIGID calculation object
rigid = RIGID(full_system=full_system)

# User defines a fragment using the molecule's coordinates
rigid.define_fragment(indices=molecule_indices, translation="xy", rotation="z")
#print error if an atom belongs to more than one fragment
#remaining atoms form a fragment that doesn't move at all (created in background)

# User sets up the ASE calculator
vasp_settings = {}
calculator = Vasp(**vasp_settings)
rigid.set_calculator(calculator)

