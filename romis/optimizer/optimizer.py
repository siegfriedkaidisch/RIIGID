class Optimizer:
    """Base class for ROMIS optimizers

    Attributes
    ----------
    optimization_history: list of romis.Optimization_Step
        The history of the optimization, which shall be checked for convergence.
    iteration: int
        Counts the number of finished optimization steps
    max_iter: int
        The maximal number of optimization steps to be performed.
        If the calculation does not converge within this limit, it is stopped.

    Note
    ----
    Depending on the optimizer, iteration and len(optimization_history) may not be the same.
    This can e.g. happen, if the optimizer drops optimization steps, where the energy grew.

    """

    def __init__(self, max_iter):
        """Initialize a ROMIS optimizer.

        Parameters
        ----------
        max_iter: int
            The maximal number of optimization steps to be performed.
            If the calculation does not converge within this limit, it is stopped.

        """
        self.optimization_history = []
        self.iteration = 0
        self.max_iter = max_iter

    def run(self, start_structure, calculator, convergence_criterion):
        """Let the optimizer run its optimization on the structure.

        Parameters
        ----------
        start_structure: romis.Structure
            The structure to be optimized
        calculator : ase.calculators.calculator.Calculator
            The used ASE calculator object
        convergence_criterion : romis.convergence.Criterion
            The used convergence criterion object

        """
        # implement restart/continue flag?
        pass

    def print_reason_for_end_of_optimization(self):
        """Prints the reason why the optimization has ended.

        Call this function when the optimization is finished.

        """
        print("Optimization finished...")
        if self.convergence_criterion.is_converged:
            print("   ...converged!")
        else:
            print("   ...NOT converged!")
            if self.iteration == self.max_iter:
                print(
                    "   Optimizer did not converge within the maximal number of allowed iterations (optimizer.max_iter)."
                )
        print()
