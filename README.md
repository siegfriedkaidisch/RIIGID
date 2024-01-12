![Logo](https://github.com/siegfriedkaidisch/ROMIS/blob/main/romis/resources/images/romis_bar.png)

ROMIS is a geometry optimization package intended to be used in theoretical solid state physics and quantum chemistry. 

The structure under investigation is separated into so-called fragments, which are a set of atoms with fixed bonds between them (usually molecules), thus forming a rigid body. Using an ASE (https://wiki.fysik.dtu.dk/ase/) calculator, the forces on each atom are found, which are then used to calculate the force and torque on each fragment. Just like rigid bodies, the fragments are then moved in accordance to these forces and torques. This way, the energy of this system of (rigid) fragments is minimized.

Author: Siegfried Kaidisch (siegfried.kaidisch(at)uni-graz.at)


![Demo Cu(O)_benzene](https://github.com/siegfriedkaidisch/ROMIS/blob/main/demo/Cu(O)_benzene/demo_CU(O)_benzene.gif)

## Quickstart

    from romis import ROMIS

    atoms = <an ASE Atoms object>
    indices = <list of indices of atoms to form a Fragment>
    name = "free to choose, e.g.: the name of the system"
    calculator_name = "the name of the calculator you want to use"
    calculator_settings = <dict of calculator settings>

    romis = ROMIS(atoms=atoms, name=name)
    romis.define_fragment_by_indices(indices=indices, allowed_translation="xyz", allowed_rotation="xyz")
    romis.set_calculator(calculator=calculator_name, settings=calculator_settings)
    romis.run()

## Installation
    
From PyPI:

    pip install romis

From GitHub:

    git clone https://github.com/siegfriedkaidisch/ROMIS
    cd ROMIS
    pip install .

## Usage
    
The user has to define the *atomic system* (as an ASE Atoms object), define the *fragments* and choose a *calculator*. Thanks to ROMIS's reliance on the ASE package, many different atomic structure codes (e.g.: VASP, https://www.vasp.at/) can be used as a calculator.

ROMIS comes with its own optimizers and convergence criteria. The user can choose explicitly which optimizer and convergence criterion to use, or keep the defaults.

Information about the optimization is written to the standard output. 
Additionally, an ASE trajectory file is produced. 

Have a look at the "demo" folder for working examples!
Note, that for the demo to work, ASE must be able to execute your VASP installation, see https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html#environment-variables.

## Known issues

For a list of known issues please see the [issues page on GitHub](https://github.com/siegfriedkaidisch/ROMIS/issues), otherwise please open a new issue.


## Misc.

The logo was created using
<a href="https://www.freepik.com/free-vector/molecular-structure-background_1476249.htm#query=molecules&position=1&from_view=search&track=sph&uuid=42cb47de-2f1f-48d1-9783-79c1c0e54126#position=1&query=molecules">this Image by katemangostar</a> on Freepik.


