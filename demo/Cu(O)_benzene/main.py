"""Example of a RIIGID geometry optimization"""

from ase.calculators.vasp.vasp import Vasp
from ase.io.vasp import read_vasp

from riigid import RIIGID
from riigid.convergence import Criterion_Displacement
from riigid.library.misc import get_atoms_indices_by_height
from riigid.optimizer import GDWAS

###############################################################################################

# Define full system
atoms = read_vasp(file="./POSCAR_start")

# Instantiate a RIIGID calculation object using an ASE atoms object of the full system
riigid = RIIGID(atoms=atoms)

# Define a fragment using the molecule's indices and define what kind of motion is allowed
# Maybe at some point additional ways of defining fragments... e.g. by adding an additional atoms object
middle_height = 9.0  # in Angstroem, used to separate molecule and surface
molecule_indices = get_atoms_indices_by_height(
    all_atoms=atoms, middle_height=middle_height, above=True, direction="z"
)
riigid.define_fragment_by_indices(
    indices=molecule_indices, allowed_translation="xy", allowed_rotation="z"
)

###############################################################################################

# Set up the ASE calculator and its settings
vasp_settings = {
    "directory": "./vasp/",
    "txt": "out",
    "npar": 4,
    "lorbit": 11,
    "lvhar": True,
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
    # The following settings should usually not be changed!
    "nsw": 0,
    "icharg": 1,
    "lcharg": True,
    "istart": 0,
    "lwave": False,
}
calculator = Vasp(**vasp_settings)
riigid.set_calculator(calculator)

###############################################################################################

# Set the RIIGID optimizer and its settings
optimizer_settings = {
    "stepsize_factor_up": 1.2,
    "stepsize_factor_dn": 0.2,
    "max_trans": 0.1,
    "max_rot": 3,
    "max_trans_0": 0.01,
    "max_rot_0": 0.3,
    "start_with_random_step": True,
    "displacement_r0": 0.001,
    "angle_r0": 0.03,
    "respect_restrictions_r0": True,
    "seed_r0": 1234,
    "max_iter": 1000,
}
optimizer = GDWAS(**optimizer_settings)  # gradient descent with adaptive stepsize
riigid.set_optimizer(optimizer)

###############################################################################################

# Set the convergence criterion
convergence_settings = {"cutoff": 0.0001}
convergence1 = Criterion_Displacement(**convergence_settings)
riigid.set_convergence_criterion(convergence1)

###############################################################################################

# Start the RIIGID optimization
riigid.run()
