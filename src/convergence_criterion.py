class Convergence_Criterion:
    """Base class for RIGID convergence criteria
    
    Attributes
    ----------
    is_converged: bool
        Whether or not the convergence criterion is fulfilled.

    """

    def __init__(self, *args, **kwargs):
        """Initialize the RIGID convergence criterion."""
        self.is_converged = False  # is the convergence criterion fulfilled?

    def check(self, optimization_history, *args, **kwargs):
        """Check if the convergence criterion is fulfilled

        If yes, self.is_converged is set to True.

        Parameters
        ----------
        optimization_history: list of optimization_step.Optimization_Step
            The history of the optimization, which shall be checked for convergence.
            (The optimization history is an attribute of the optimizer.)

        """
        pass
