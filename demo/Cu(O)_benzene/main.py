"""Example of a RIGID geometry optimization

"""

from ase.calculators.vasp.vasp import Vasp
from ase.io.vasp import read_vasp

from rigid import RIGID
from rigid.convergence import CC_Displacement
from rigid.library import get_atoms_indices_by_height
from rigid.optimizer import GDWAS

# User defines full system
atoms = read_vasp(file="./POSCAR_start")

# User instantiates a RIGID calculation object using an ASE atoms object of the full system
rigid = RIGID(atoms=atoms, name="example_system")

# User defines a fragment using the molecule's coordinates and defines what kind of motion is allowed
# Maybe at some point additional ways of defining fragments... e.g. by adding an additional atoms object
middle_height = 9.0  # in Angstroem, used to separate molecule and surface
molecule_indices = get_atoms_indices_by_height(
    all_atoms=atoms, middle_height=middle_height, above=True, direction="z"
)
rigid.define_fragment_by_indices(
    indices=molecule_indices, allowed_translation="xy", allowed_rotation="z"
)

# User sets up the ASE calculator and its settings
vasp_settings = {
    "directory": "./vasp/",
    "txt": "out",
    "istart": 0,
    "npar": 4,
    "lorbit": 11,
    "lvhar": True,
    "lwave": True,
    "prec": "Accu",
    "encut": 400,
    "ediff": 1e-7,
    "nelmin": 4,
    "nelm": 200,
    "algo": "Fast",
    #    'enaug':644.9,
    "xc": "PBE",
    "ivdw": 12,
    "ismear": 0,
    "sigma": 0.05,
    "idipol": 3,
    "ldipol": True,
    #    'ispin':2,
    #    'magmom':[0.0]*224 + [1.5],
    #    'icharg':1,
    "nsw": 0,  # leave this at zero!
    #    'ldau':True,
    #    'ldautype':2,
    #    'ldaul':[-1]*5 + [2],
    #    'ldauu':[0.0]*5 + [3.0],
    #    'ldauj':[0.0]*6,
    #    'ldauprint':2,
    #    'lmaxmix':4,
    "kpts": [2, 2, 1],
    "gamma": True,  # Gamma-centered k-mesh
    "setups": "recommended",
}
calculator = Vasp(**vasp_settings)
rigid.set_calculator(calculator)

# User sets the RIGID optimizer and its settings
optimizer_settings = {
    "stepsize_factor_up": 1.2,
    "stepsize_factor_dn": 0.2,
    "max_step": 0.1,
    "max_step_0": 0.01,
    "start_with_random_step": True,
    "displacement_r0": 0.01,
    "angle_r0": 0.1,
    "respect_restrictions_r0": False,
    "seed_r0": 1234,
    "max_iter": 100
}
optimizer = GDWAS(**optimizer_settings)  # gradient descent with adaptive stepsize
rigid.set_optimizer(optimizer)

# User sets the convergence criterion
# Would be cool, if user could set multiple criteria!
convergence_settings = {"cutoff": 0.0001}
convergence1 = CC_Displacement(**convergence_settings)
rigid.set_convergence_criterion(convergence1)

# Start the rigid optimization
# add verbosity settings?
# add control over output?
# if so, do this here, or at init of RIGID object?
rigid.run()
