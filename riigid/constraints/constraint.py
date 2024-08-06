class Constraint:
    """Template for constraints."""

    def __init__(self):
        pass

    def init_internally(self, start_structure):
        """Internal init of the class.

        This can be used to initialize some class attribute that depend on the start-structure.

        Parameters
        ----------
        start_structure : riigid.Structure
            The full (start-)structure of the calculation.

        """
        pass

    def check(self, structure, cutoff):
        """Check if the constraint is fulfilled currently.

        Parameters
        ----------
        structure: riigid.Structure
            The structure to be checked for the constraint.
        cutoff:
            Some cutoff value, that defines if the constraint is fulfilled or not.
            A reasonable default should be set.

        Returns
        -------
        bool
            Is the constraint fulfilled?

        """
        pass

    def apply(self, structure):
        """Apply the constraint.

        Parameters
        ----------
        structure: riigid.Structure
            The constraint is applied to this supplied structure.

        """
        pass


class List_of_Contraints:
    """List of constrains.

    If you have more that one constraints, this can be used to apply
    them all together on a structure.
    This can be useful if constraints "work against each other". This class
    implements methods that make sure that all constraints are fulfilled.

    Attributes
    ----------
    constraints: list of riigid.constraints.constraint.Constraint
        The constraints to be applied, collected in a list.
    """

    def __init__(self):
        self.constraints = []

    def add(self, constraint):
        """Add a constraint to the list"""
        self.constraints += [constraint]

    def init_internally(self, start_structure):
        """Call internal init of all constraints

        Parameters
        ----------
        start_structure : riigid.Structure
            The full (start-)structure of the calculation.

        """
        for constraint in self.constraints:
            constraint.init_internally(start_structure=start_structure)

    def check(self, structure):
        """Check if the constraints are fulfilled currently.

        Parameters
        ----------
        structure: riigid.Structure
            The structure to be checked for the constraints.

        Returns
        -------
        bool
            Are all constraints fulfilled?

        """
        for constraint in self.constraints:
            if not constraint.check(structure=structure):
                return False
        return True

    def apply(self, structure, num_it=1000):
        """Apply all constraints

        Iteratively apply all constraints until they are all fulfilled.

        Parameters
        ----------
        structure: riigid.Structure
            The constraints are applied to this supplied structure.
        num_it: int, optional, default is 1000
            If after `num_it` applications of all constraints, they are still
            not all fulfilled, an exception is thrown.

        """
        counter = 0
        while counter < num_it:
            counter += 1

            for constraint in self.constraints:
                constraint.apply(structure=structure)
            converged = self.check(structure=structure)
            if converged:
                return
        else:
            raise Exception("Constraints did not converge!")
