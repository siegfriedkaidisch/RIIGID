import numpy as np

from riigid.convergence.criterion import Criterion


class Criterion_Force_Torque(Criterion):
    """RIIGID convergence criterion: Force and Torque

    If max force and torque on fragments are below cutoffs, the criterion is fulfilled.

    Attributes
    ----------
    is_converged: bool
        Whether or not the convergence criterion is fulfilled.
    cutoff_f, cutoff_t: float, float
        Cutoffs for force and torque; [eV/Å], [eV]

    """

    def __init__(self, cutoff_f=0.01, cutoff_t=0.01):
        """Initialize the Force+Torque convergence criterion.

        Parameters
        ----------
        cutoff_f, cutoff_t: float, float, both optional, defaults: 0.01, 0.01
            Cutoffs for force and torque; [eV/Å], [eV]

        """
        super().__init__()
        self.cutoff_f = cutoff_f
        self.cutoff_t = cutoff_t

    def check(self, optimization_history):
        """Check if the convergence criterion is fulfilled.

        If yes, self.is_converged is set to True.

        Parameters
        ----------
        optimization_history: list of riigid.Optimization_Step
            The history of the optimization, which shall be checked for convergence.
            (The optimization history is an attribute of the optimizer.)

        """
        # Get max force and torque on fragments from last optimization step
        last_step = optimization_history[-1]
        max_f = last_step.max_force_on_fragment
        max_t = last_step.max_torque_on_fragment

        if (max_f < self.cutoff_f) and (max_t < self.cutoff_t):
            self.is_converged = True
