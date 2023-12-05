from copy import deepcopy
import numpy as np

from optimization_step import Optimization_Step

class Optimizer():
    '''
    No direct interaction with user. That should be done via RIGID class
    '''
    def __init__(self, *args, **kwargs):
        self.optimization_history = []
        self.iteration = 0

    def run(self, structure, calculator, convergence_criterion, *args, **kwargs):
        pass

class GDWALR(Optimizer):
    '''
    Gradient Descent with Adaptive Learning Rate
    '''
    def __init__(self, stepsize_factor_up, stepsize_factor_dn, max_step, max_step_0, *args, **kwargs):
        super().__init__()
        self.stepsize = 100 # initial stepsize doesn't matter
        self.stepsize_factor_up = stepsize_factor_up
        self.stepsize_factor_dn = stepsize_factor_dn
        self.max_step = max_step
        self.max_step_0 = max_step_0

    def run(self, start_structure, calculator, convergence_criterion, *args, **kwargs):
        while not convergence_criterion.is_converged:
            if self.iteration == 0:
                structure = start_structure
            else:
                structure = self.optimization_history[-1].updated_structure
            
            # Do Calculation 
            energy, forces = structure.calculate_energy_and_forces(calculator)

            # Adapt stepsize to energy change
            if self.iteration > 0: 
                self.adapt_stepsize_to_energy_change(old_energy=self.optimization_history[-1].energy, new_energy=energy)

            # Initialize next optimization step (undo last update if energy got larger)
            self.initialize_next_optimization_step(structure, forces, energy)
            structure = self.optimization_history[-1].structure
            forces = self.optimization_history[-1].forces
            energy = self.optimization_history[-1].energy

            # Adapt stepsize again, if necessary (do test step to prevent too large movement)
            self.adapt_stepsize_to_prevent_too_large_steps(structure, forces)

            # Move atoms
            _, _, updated_structure = structure.move(self.stepsize, forces, change=False)#don't change in place
            self.optimization_history[-1].add_updated_structure(updated_structure)

            # Prepare next iteration
            self.iteration +=1

            # Check for convergence
            self.convergence_criterion.check(self.optimization_history[-1])

    def adapt_stepsize_to_energy_change(self, old_energy, new_energy):
        # Check if energy got smaller
        if new_energy < old_energy: # Yes -> make stepsize larger
            self.stepsize *= self.stepsize_factor_up
        else: # No -> make stepsize smaller and drop latest optimization step
            self.stepsize *= self.stepsize_factor_dn

    def adapt_stepsize_to_prevent_too_large_steps(self, structure, forces):
        positions_difference_test_max, _, _ = structure.move(self.stepsize, forces, change=False)

        # Initialize stepsize or adapt it to too large atomic displacements
        if self.iteration == 0: # In the first iteration we determine the initial stepsize, such that a step of max_step_0 is performed
            factor = self.max_step_0/positions_difference_test_max
        else: # In later iterations, calc factor to prevent too large atomic deplacements
            if positions_difference_test_max>self.max_step:
                factor = self.max_step/positions_difference_test_max
            else:
                factor = 1.0
        self.stepsize *= factor

    def initialize_next_optimization_step(self, structure, forces, energy):
        if self.iteration == 0:
            self.optimization_history.append(Optimization_Step(structure, forces, energy))
        else:
            prev_energy = self.optimization_history[-1].energy
            # Check if energy got smaller
            if energy < prev_energy: # Yes 
                self.optimization_history.append(Optimization_Step(structure, forces, energy))
            else: # No ->  drop latest optimization step
                self.optimization_history[-1].remove_updated_structure()





        





        
                
