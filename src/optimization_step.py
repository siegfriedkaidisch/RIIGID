class Optimization_Step():
    '''
    Data-class, i.e. only to store data
    '''
    def __init__(self, structure, forces, energy, updated_structure=None):
        self.structure = structure
        self.forces = forces
        self.energy = energy
        self.updated_structure = updated_structure

    def add_updated_structure(self, updated_structure):
        self.updated_structure = updated_structure

    def remove_updated_structure(self):
        self.updated_structure = None
