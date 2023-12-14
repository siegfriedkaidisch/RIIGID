class Optimizer():
    """
    Base class for RIGID optimizers
    """
    def __init__(self, *args, **kwargs):
        self.optimization_history = [] # list of optimization steps
        self.iteration = 0 # number of finished optimization steps

    def run(self, start_structure, calculator, convergence_criterion, *args, **kwargs):
        """
        Let the optimizer run its optimization on the structure

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



