![Logo](https://github.com/siegfriedkaidisch/RIIGID/blob/main/riigid/resources/images/riigid_bar.png)

RIIGID.py is a geometry optimization package intended to be used in theoretical solid state physics and quantum chemistry. 

The structure under investigation is separated into so-called fragments, which are a set of atoms with fixed bonds between them (usually molecules), thus forming a rigid body. Using an [ASE](https://wiki.fysik.dtu.dk/ase/) calculator, the forces on each atom are found, which are then used to calculate the force and torque on each fragment. Just like rigid bodies, the fragments are then moved in accordance to these forces and torques. This way, the energy of this system of (rigid) fragments is minimized.

[Read the Docs](https://riigid.readthedocs.io/en/latest/index.html)

Author: Siegfried Kaidisch (siegfried.kaidisch(at)uni-graz.at)


![Demo Cu(O)_benzene](https://github.com/siegfriedkaidisch/RIIGID/blob/main/demo/Cu(O)_benzene/demo_CU(O)_benzene.gif)

## Quickstart

```python
from riigid import RIIGID

atoms = <an ASE Atoms object>
indices = <list of indices of atoms to form a Fragment>
name = "free to choose, e.g.: the name of the system"
calculator_name = "the name of the calculator you want to use"
calculator_settings = <dict of calculator settings>

riigid = RIIGID(atoms=atoms, name=name)
riigid.define_fragment_by_indices(indices=indices, allowed_translation="xyz", allowed_rotation="xyz")
riigid.set_calculator(calculator=calculator_name, settings=calculator_settings)
riigid.run()
```

## Installation
    
From PyPI:

    pip install RIIGID

From GitHub:

    git clone https://github.com/siegfriedkaidisch/RIIGID
    cd RIIGID
    pip install .

## Usage
    
The user has to define the *atomic system* (as an ASE Atoms object), define the *fragments* and choose a *calculator*. Thanks to RIIGID's reliance on the ASE package, many different atomic structure codes (e.g.: [VASP](https://www.vasp.at/)) can be used as a calculator.

RIIGID comes with its own optimizers and convergence criteria. The user can choose explicitly which optimizer and convergence criterion to use, or keep the defaults.

Information about the optimization is written to the standard output. 
Additionally, an ASE trajectory file is produced. 

Have a look at the "demo" folder for working examples!
Note, that for the demo to work, ASE must be able to execute your VASP installation, see the [ASE documentation](https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html#environment-variables).

## Known issues

For a list of known issues please see the [issues page on GitHub](https://github.com/siegfriedkaidisch/RIIGID/issues), otherwise please open a new issue.


## Misc.

The logo was created using
<a href="https://www.freepik.com/free-vector/molecular-structure-background_1476249.htm#query=molecules&position=1&from_view=search&track=sph&uuid=42cb47de-2f1f-48d1-9783-79c1c0e54126#position=1&query=molecules">this Image by katemangostar</a> on Freepik.


