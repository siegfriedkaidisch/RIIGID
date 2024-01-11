# RIGID.py

RIGID.py is a geometry optimization package intended to be used in theoretical solid state physics and quantum chemistry. 

The structure under investigation is separated into so-called fragments, which are a set of atoms with fixed bonds between them, thus forming a rigid body. Using an ASE (https://wiki.fysik.dtu.dk/ase/) calculator, the forces on each atom are found, which are then used to calculate the force and torque on each fragment. Just like rigid bodies, the fragments are then moved in accordance to these forces and torques (like rigid bodies). This way, the energy of this system of (rigid) fragments is minimized.

![Test](https://github.com/siegfriedkaidisch/RIGID.py/blob/main/demo/Cu(O)_benzene/demo_CU(O)_benzene.gif)


Author: Siegfried Kaidisch (siegfried.kaidisch(at)uni-graz.at)

## Quickstart

    pip install rigid.py

## Installation
    
From PyPI:

    pip install rigid.py

From GitHub:

    git clone https://github.com/siegfriedkaidisch/RIGID.py
    cd RIGID.py
    pip install .

## Usage
    
The user has to define the *atomic system* (as an ASE Atoms object), define the *fragments* and choose a *calculator*. Thanks to RIGID's reliance on the ASE package, many different atomic structure codes (e.g.: VASP, https://www.vasp.at/) can be used as a calculator.

RIGID comes with its own optimizers and convergence criteria. The user can choose explicitly which optimizer and convergence criterion to use, or keep the defaults.

Information about the optimization is written to the standard output. 
Additionally, and ASE trajectory file is produced. 

Have a look at the "demo" folder for working examples!
Note, that for the demo to work, ASE must be able to execute your VASP installation, see https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html#environment-variables.

## Known issues

For a list of known issues please see the [issues page on GitHub](https://github.com/siegfriedkaidisch/RIGID.py/issues), otherwise please open a new issue.