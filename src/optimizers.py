from copy import deepcopy

from optimization_step import Optimization_Step

class Optimizer():
    '''
    No direct interaction with user. That should be done via RIGID class
    '''
    def __init__(self, *args, **kwargs):
        self.optimization_history = []
        self.iteration = 0

    def run(self, start_structure, calculator, convergence_criterion, *args, **kwargs):
        # restart/continue flag?
        pass

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
            if self.iteration == 0:
                self.current_structure = deepcopy(start_structure)
            else:
                self.current_structure = deepcopy(self.optimization_history[-1].updated_structure)
            
            # Do Calculation 
            self.current_energy, self.current_forces = self.current_structure.calculate_energy_and_forces(calculator=calculator)

            # Adapt stepsize to energy change
            if self.iteration > 0: 
                self.adapt_stepsize_to_energy_change(old_energy=self.optimization_history[-1].energy, new_energy=self.current_energy)

            # Initialize next optimization step (undo last update if energy got larger)
            self.initialize_next_optimization_step()
            
            # Adapt stepsize again, if necessary (do test step to prevent too large movement)
            test_structure = deepcopy(self.current_structure)
            max_atomic_displacement_test, _ = test_structure.move(forces=self.current_forces, stepsize=self.stepsize)
            self.adapt_stepsize_to_prevent_too_large_steps(max_atomic_displacement=max_atomic_displacement_test)

            # Move atoms
            updated_structure = deepcopy(self.current_structure)
            _, _ = updated_structure.move(forces=self.current_forces, stepsize=self.stepsize)
            self.optimization_history[-1].add_updated_structure(updated_structure=updated_structure)

            # Prepare next iteration
            self.iteration +=1

            # Check for convergence
            self.convergence_criterion.check(optimization_history=self.optimization_history)

    def adapt_stepsize_to_energy_change(self, old_energy, new_energy):
        # Check if energy got smaller
        if new_energy < old_energy: # Yes -> make stepsize larger
            self.stepsize *= self.stepsize_factor_up
        else: # No -> make stepsize smaller and drop latest optimization step
            self.stepsize *= self.stepsize_factor_dn

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

    def initialize_next_optimization_step(self):
        structure = self.current_structure
        forces    = self.current_forces
        energy    = self.current_energy

        if self.iteration == 0:
            self.optimization_history.append(Optimization_Step(structure=structure, forces=forces, energy=energy))
        else:
            prev_energy = self.optimization_history[-1].energy
            # Check if energy got smaller
            if energy < prev_energy: # Yes 
                self.optimization_history.append(Optimization_Step(structure=structure, forces=forces, energy=energy))
            else: # No ->  drop latest optimization step
                self.optimization_history[-1].remove_updated_structure()
                self.current_structure = self.optimization_history[-1].structure
                self.current_forces = self.optimization_history[-1].forces
                self.current_energy = self.optimization_history[-1].energy


