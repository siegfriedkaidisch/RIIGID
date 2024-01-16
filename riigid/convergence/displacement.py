import numpy as np

from riigid.convergence.criterion import Criterion


class Criterion_Displacement(Criterion):
    """RIIGID convergence criterion: Displacement

    If all atoms move less than cutoff, the criterion is fulfilled.

    Note
    ----
    This criterion could theoretically be fulfilled, even if the calculation is not actually
    converged. This is, because the displacement of the atoms not only depends on the force and torque
    acting on each fragment, but also on the stepsize. If the stepsize is very small, the atoms may move
    only very little, even though the forces and torques are still large. If you set the stepsize by hand,
    make sure not to choose a too small value, or use a different/additional convergence criterion. If you
    use an optimizer with automatic stepsize, the optimizer should prevent this from happening.

    Attributes
    ----------
    is_converged: bool
        Whether or not the convergence criterion is fulfilled.
    cutoff: number
        If all atoms move less than cutoff, the criterion is fulfilled; [Å]

    """

    def __init__(self, cutoff=1e-6):
        """Initialize the Displacement convergence criterion.

        Parameters
        ----------
        cutoff: number, default:1e-6
            If all atoms move less than cutoff, the criterion is fulfilled; [Å]

        """
        super().__init__()
        self.cutoff = cutoff

    def check(self, optimization_history):
        """Check if the convergence criterion is fulfilled.

        If yes, self.is_converged is set to True.

        Parameters
        ----------
        optimization_history: list of riigid.Optimization_Step
            The history of the optimization, which shall be checked for convergence.
            (The optimization history is an attribute of the optimizer.)

        """
        # Compare initial and updated structure from last optimization step
        last_step = optimization_history[-1]
        old_positions = last_step.structure.atoms.positions
        new_positions = last_step.updated_structure.atoms.positions

        # Scalar displacement of all individual atoms
        displacements = np.linalg.norm(new_positions - old_positions, axis=1)

        # Maximal scalar displacement of an atom
        max_displacement = np.max(displacements)

        if max_displacement < self.cutoff:
            self.is_converged = True
