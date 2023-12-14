import numpy as np

from convergence_criterion import Convergence_Criterion

class CC_Displacement(Convergence_Criterion):
    """
    RIGID convergence criterion: Displacement

    If all atoms move less than cutoff, the criterion is fulfilled.

    Note, that this criterion could theoretically be fulfilled, even if the calculation is not actually
    converged. This is, because the displacement of the atoms not only depends on the force and torque 
    acting on each fragment, but also on the stepsize. If the stepsize is very small, the atoms may move 
    only very little, even though the forces and torques are still large. If you set the stepsize by hand,
    make sure not to choose a too small value, or use a different/additional convergence criterion. If you 
    use an optimizer with automatic stepsize, the optimizer should prevent this from happening.
    """

    def __init__(self, cutoff, *args, **kwargs):
        """
        Parameters
        ----------
        cutoff: number
            If all atoms move less than cutoff, the criterion is fulfilled; [AA]
        """
        super().__init__()
        self.cutoff = cutoff

    def check(self, optimization_history, *args, **kwargs):
        """
        Check if the convergence criterion is fulfilled

        Parameters
        ----------
        optimization_history: list of optimization_step.Optimization_Step
            The history of the optimization, which shall be checked for convergence. 
            (The optimization history is an attribute of the optimizer.)
        """
        last_step = optimization_history[-1]
        old_positions = last_step.structure.atoms.positions
        new_positions = last_step.updated_structure.atoms.positions

        displacements  = np.linalg.norm(new_positions - old_positions, axis=1) 
        max_displacement = np.max(displacements)

        if max_displacement < self.cutoff:
            self.is_converged = True

            