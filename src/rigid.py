class RIGID():
    '''
    I/O class, user should ideally only interact with this class
    '''
    def __init__(self, atoms):
        self.structure = Structure(atoms=atoms)

    def define_fragment(self, *args, **kwargs):
        self.structure.define_fragment(*args, **kwargs)

    def set_calculator(self, calculator):
        self.calculator = calculator

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_convergence_criterion(self, convergence_criterion):
        self.convergence_criterion = convergence_criterion

    def run(self):
        # Perform rigid optimization
        self.optimizer.run(start_structure=self.structure, calculator=self.calculator, 
                           convergence_criterion=self.convergence_criterion)
        
        # Save and print results
        self.save_optimization_data() # get raw data from optimizer
        self.print_optimization_summary() # get raw data from optimizer

    def save_optimization_data(self):
        pass

    def print_optimization_summary(self):
        pass



class Structure:
    def move(forces):
        for i, fragment in enumerate(self.framgnets):
            fragmnet.move(forces[self.indcies[i]])

    def calculate_energy_and_forces(self, calculator):
        # better: method of Structure?
        atoms = self.atoms
        atoms.set_calculator(calculator)
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        return energy, forces