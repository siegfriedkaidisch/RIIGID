import numpy as np

from convergence_criterion import Convergence_Criterion

class CC_Displacement(Convergence_Criterion):
    '''
    Hello
    '''
    def __init__(self, cutoff, *args, **kwargs):
        super().__init__()
        self.cutoff = cutoff

    def check(self, optimization_history, *args, **kwargs):
        last_step = optimization_history[-1]
        old_positions = last_step.structure.atoms.positions
        new_positions = last_step.updated_structure.atoms.positions

        displacements  = np.linalg.norm(new_positions - old_positions, axis=1) 
        max_displacement = np.max(displacements)

        if max_displacement < self.cutoff:
            self.is_converged = True

            