from copy import deepcopy


class Criterion:
    """Base class for RIIGID convergence criteria

    Attributes
    ----------
    is_converged: bool
        Whether or not the convergence criterion is fulfilled.

    """

    def __init__(self):
        """Initialize the RIIGID convergence criterion."""
        self.is_converged = False  # initialize as unfulfilled

    def check(self, optimization_history):
        """Check if the convergence criterion is fulfilled

        If yes, self.is_converged is set to True.

        Parameters
        ----------
        optimization_history: list of riigid.Optimization_Step
            The history of the optimization, which shall be checked for convergence.
            (The optimization history is an attribute of the optimizer.)

        """
        pass

    def __and__(self, cc2):
        """Return a new convergence criterion (CC), as logical "and" of two CCs.

        Usage: cc1_and_cc2 = cc1 & cc2

        Parameters
        ----------
        cc2: riigid.convergence.Criterion
            The second convergence criterion.

        Returns
        -------
        riigid.convergence.Criterion
            The logical "and" of self and cc1

        """
        return Compound_Criterion(cc1=deepcopy(self), cc2=deepcopy(cc2), operator="and")

    def __or__(self, cc2):
        """Return a new convergence criterion (CC), as logical "or" of two CCs.

        Usage: cc1_or_cc2 = cc1 | cc2

        Parameters
        ----------
        cc2: riigid.convergence.Criterion
            The second convergence criterion.

        Returns
        -------
        riigid.convergence.Criterion
            The logical "or" of self and cc1

        """
        return Compound_Criterion(cc1=deepcopy(self), cc2=deepcopy(cc2), operator="or")


class Compound_Criterion(Criterion):
    """Compound convergence criterion.

    Logical combination of two convergence criteria.

    Attributes
    ----------
    is_converged: bool
        Whether or not the convergence criterion is fulfilled.
    cc1, cc2: riigid.convergence.Criterion
        The two convergence criteria to combine.
    operator: str, options: "and", "or"
        How shall the two convergence criteria be combined?

    """

    def __init__(self, cc1, cc2, operator):
        """Initialize the combined convergence criterion.

        Parameters
        ----------
        cc1, cc2: riigid.convergence.Criterion
            The two convergence criteria to combine.
        operator: str, options: "and", "or"
            How shall the two convergence criteria be combined?

        """
        super().__init__()
        self.cc1 = cc1
        self.cc2 = cc2
        self.operator = operator

    def check(self, optimization_history):
        """Check if the convergence criterion is fulfilled

        If yes, self.is_converged is set to True.

        Parameters
        ----------
        optimization_history: list of riigid.Optimization_Step
            The history of the optimization, which shall be checked for convergence.
            (The optimization history is an attribute of the optimizer.)

        """
        # Check the two individual CCs
        self.cc1.check(optimization_history=optimization_history)
        self.cc2.check(optimization_history=optimization_history)

        # Check the compound CC
        if self.operator == "and":
            if self.cc1.is_converged and self.cc2.is_converged:
                self.is_converged
        elif self.operator == "or":
            if self.cc1.is_converged or self.cc2.is_converged:
                self.is_converged
        else:
            raise Exception(
                "This operator is not a known operator to combine convergence criteria! Options:'and', 'or'"
            )
