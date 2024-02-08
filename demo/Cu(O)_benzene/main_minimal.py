"""Minimal example of a RIIGID geometry optimization

"""

from ase.io.vasp import read_vasp

from riigid import RIIGID
from riigid.library.misc import get_atoms_indices_by_height

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
riigid.set_calculator(calculator="Vasp", settings=vasp_settings)

###############################################################################################

# Start the RIIGID optimization
riigid.run()
