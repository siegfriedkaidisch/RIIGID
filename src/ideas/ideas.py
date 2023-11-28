from RIGID import *

###rigid = RIGID(Path (fullstructure) | Atoms)
###fragment = rigid.define_fragment(list of indicies)
###fragment.set_constraint("Rx", vary=True)
rigid.vasp_settings({})
rigid.set_convergence()


# RIGID.add_substrate(Path | Atoms)
# RIGID.add_fragment(Path | Atoms)

# RIGID.set_structure(Path)
# RIGID.set_fragmnet(indices)

RIGID.run()


class RIGID():
    def __init__(self):
        self.structure = Structure()
        self.vasp_settings

    def define_fragemnet(*args, **kwargs):
        self.structure.define_fragemnet(*args, **kwargs)

    def run():
        while not self.critera.isConverged():
            self.run_vasp()

        
        self.output()

class UpdateStep():

    def __init__(self, structure, forces):
        self.old_structure = structure
        self.forces = forces
        self.new_structure = self.apply()

    def apply():
        self.old_strucutre.move(forces)
    
    def step():
        self.calculation_forces()
        self.update_position()

class Structure:

    def move(forces):
        for i, fragment in enumerate(self.framgnets):
            fragmnet.move(forces[self.indcies[i]])


name/name|src/*.py
name/setup.py
name/Makefile