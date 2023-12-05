class Structure():
    def __init__(self, atoms):
        self.atoms = atoms

    def define_fragment(self, indices, translation, rotation):
        pass

    def calculate_energy_and_forces(self, calculator):
        pass

    def move(self, forces, stepsize, change):
        for i, fragment in enumerate(self.framgnets):
            fragmnet.move(forces[self.indcies[i]])







    def calculate_energy_and_forces(self, calculator):
        # better: method of Structure?
        atoms = self.atoms
        atoms.set_calculator(calculator)
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        return energy, forces