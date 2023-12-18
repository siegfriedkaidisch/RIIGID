class Optimizer():
    """Base class for RIGID optimizers

    Attributes
    ----------
    optimization_history: list of optimization_step.Optimization_Step
        The history of the optimization, which shall be checked for convergence. 
    iteration: int
        Counts the number of finished optimization steps

    Note
    ----
    Depending on the optimizer, iteration and len(optimization_history) may not be the same. 
    This can e.g. happen, if the optimizer drops optimization steps, where the energy grew.
    
    """
    def __init__(self, *args, **kwargs):
        """Initialize a RIGID optimizer.
        
        """
        self.optimization_history = [] # list of optimization steps
        self.iteration = 0 # number of finished optimization steps

    def run(self, start_structure, calculator, convergence_criterion, *args, **kwargs):
        """Let the optimizer run its optimization on the structure.

        Parameters
        ----------
        start_structure: structure.Structure
            The structure to be optimized
        calculator : ase.calculators.calculator.Calculator
            The used ASE calculator object
        convergence_criterion : convergence_criterion.Convergence_Criterion
            The used convergence criterion object

        """
        # implement restart/continue flag?
        pass



