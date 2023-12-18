"""Example of a RIGID geometry optimization

"""

from ase import Atoms
from ase.calculators.vasp.vasp import Vasp

from convergence_criterion import Convergence1
from optimizer import Optimizer
from rigid import RIGID

# User defines full system
atoms = Atoms()

# User instantiates a RIGID calculation object using an ASE atoms object of the full system
rigid = RIGID(atoms=atoms)

# User defines a fragment using the molecule's coordinates and defines what kind of motion is allowed
molecule_indices = [1, 2, 3]
rigid.define_fragment_by_indices(
    indices=molecule_indices, allowed_translation="xy", allowed_rotation="z"
)
# print error if an atom belongs to more than one fragment
# remaining atoms form a fragment that doesn't move at all (created in background)

# User sets up the ASE calculator and its settings
vasp_settings = {}
calculator = Vasp(**vasp_settings)
rigid.set_calculator(calculator)

# User sets the RIGID optimizer and its settings
rigid_settings = {}
optimizer = Optimizer(**rigid_settings)  # gradient descent with adaptive stepsize
rigid.set_optimizer(optimizer)

# User sets the convergence criterion
# Would be cool, if user could set multiple criteria!
convergence1_settings = {}
convergence1 = Convergence1(**convergence1_settings)
rigid.set_convergence_criterion(convergence1)

# Start the rigid optimization
# add verbosity settings?
# add control over output?
# if so, do this here, or at init of RIGID object
rigid.run()
