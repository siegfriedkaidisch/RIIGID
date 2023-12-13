from copy import deepcopy, copy

from optimization_step import Optimization_Step
from optimizers import Optimizer

class GDWALR(Optimizer):
    '''
    Gradient Descent with Adaptive Learning Rate
    '''
    def __init__(self, stepsize_factor_up, stepsize_factor_dn, max_step, max_step_0, *args, **kwargs):
        super().__init__()
        self.stepsize = 100 # initial stepsize, value doesn't matter
        self.stepsize_factor_up = stepsize_factor_up
        self.stepsize_factor_dn = stepsize_factor_dn
        self.max_step = max_step
        self.max_step_0 = max_step_0

    def run(self, start_structure, calculator, convergence_criterion, *args, **kwargs):
        self.start_structure = start_structure
        self.calculator = calculator
        self.convergence_criterion = convergence_criterion

        while not convergence_criterion.is_converged:
            # Get current structure
            if self.iteration == 0:
                self.current_structure = deepcopy(start_structure)
            else:
                self.current_structure = deepcopy(self.optimization_history[-1].updated_structure)
            
            # Do Calculation 
            self.current_energy, self.current_forces = self.current_structure.calculate_energy_and_forces(calculator=calculator)

            # Adapt stepsize to energy change
            self.adapt_stepsize_to_energy_change()

            # undo last update if energy got larger
            self.drop_last_step_if_energy_got_larger()
            
            # Adapt stepsize again, if necessary (do test step to prevent too large movement)
            test_structure = deepcopy(self.current_structure)
            max_atomic_displacement_test, _ = test_structure.move(forces=self.current_forces, stepsize=self.stepsize)
            self.adapt_stepsize_to_prevent_too_large_steps(max_atomic_displacement=max_atomic_displacement_test)

            # Move atoms
            updated_structure = deepcopy(self.current_structure)
            _, _ = updated_structure.move(forces=self.current_forces, stepsize=self.stepsize)

            # Add Optimization step to history
            self.add_new_optimization_step_to_history(structure=self.current_structure, 
                                                      forces=self.current_forces, 
                                                      energy=self.current_energy, 
                                                      updated_structure=updated_structure)

            # Prepare next iteration
            self.iteration +=1

            # Check for convergence
            self.convergence_criterion.check(optimization_history=self.optimization_history)

    def adapt_stepsize_to_energy_change(self):
        if self.iteration != 0:
            prev_energy    = self.optimization_history[-1].energy
            current_energy = self.current_energy
            # Check if energy got smaller
            if current_energy < prev_energy: # Yes -> make stepsize larger
                self.stepsize *= self.stepsize_factor_up
            else: # No -> make stepsize smaller and drop latest optimization step
                self.stepsize *= self.stepsize_factor_dn

    def drop_last_step_if_energy_got_larger(self):
        if self.iteration != 0:
            prev_energy    = self.optimization_history[-1].energy
            current_energy = self.current_energy
            if prev_energy < current_energy:
                self.current_structure = deepcopy(self.optimization_history[-1].structure)
                self.current_forces = copy(self.optimization_history[-1].forces)
                self.current_energy = copy(self.optimization_history[-1].energy)
                self.optimization_history.pop()

    def adapt_stepsize_to_prevent_too_large_steps(self, max_atomic_displacement):
        # Initialize stepsize or adapt it to too large atomic displacements
        if self.iteration == 0: # In the first iteration we determine the initial stepsize, such that a step of max_step_0 is performed
            factor = self.max_step_0/max_atomic_displacement
        else: # In later iterations, calc factor to prevent too large atomic deplacements
            if max_atomic_displacement>self.max_step:
                factor = self.max_step/max_atomic_displacement
            else:
                factor = 1.0
        self.stepsize *= factor

    def add_new_optimization_step_to_history(self, *args, **kwargs):
        self.optimization_history.append(Optimization_Step(*args, **kwargs))
