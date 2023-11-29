class RIGID():
    def __init__(self, full_system):
        self.structure = Structure(full_system=full_system)

    def define_fragment(self, *args, **kwargs):
        self.structure.define_fragment(*args, **kwargs)

    def set_calculator(self, calculator):
        self.calculator = calculator

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_convergence_criterion(self, convergence_criterion):
        self.convergence_criterion = convergence_criterion

    def run(self):
        while not self.convergence_criterion.is_converged:
            
            # Do Calculation 
            current_structure = self.optimizer.optimization_steps[-1].new_structure
            current_full_system = current_structure.full_system
            current_full_system.set_calculator(self.calculator)
            current_energy = current_full_system.get_potential_energy()
            current_forces = current_full_system.get_forces()

            # Do Geometry Optimization Step
            optimization_step = self.optimizer.do_step(current_energy, current_forces)

            # Check for Convergence
            self.convergence_criterion.check(optimization_step)
        
        # Finish the Optimization
        self.save_optimization_data()
        self.print_optimization_summary()

class Optimization_Step():

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