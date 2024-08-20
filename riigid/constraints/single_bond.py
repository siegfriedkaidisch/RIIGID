import numpy as np

from riigid.constraints.constraint import Constraint


class Single_Bond(Constraint):
    """Constraint: Take one atom per fragment and freeze the bond length between them.

    Example: Separate 6T into six different fragments and fix the bond length between
    neighboring tiophenes.

    Attributes
    ----------
    iatom1, iatom2: int, int
        The indices of the two relevant atoms.
        The distance between these two atoms will be kept constant.
        They must be parts of different fragments.
        Indices are defined relative to the full Structure's `atoms` attribute.
    _desired_bondlength: float
        Just for internal use!
        Stores the length of the bond, before the first geometry step; [Å]

    """

    def __init__(self, iatom1, iatom2):
        self.iatom1 = iatom1
        self.iatom2 = iatom2

    def init_internally(self, start_structure):
        """Internal init of the class.

        Calculates the bond length of the start-structure.
        This must be called before the first optimization step is done!

        Parameters
        ----------
        start_structure : riigid.Structure
            The full (start-)structure of the calculation.

        """
        pos1 = start_structure.atoms.positions[self.iatom1]
        pos2 = start_structure.atoms.positions[self.iatom2]
        self._desired_bondlength = np.linalg.norm(pos2 - pos1)

    def check(self, structure, cutoff=1e-6):
        """Check the constraint.

        Parameters
        ----------
        structure: riigid.Structure
            The structure to be checked for the constraint.
        cutoff: float, optional, default is 1e-6
            If |1 - current_bondlength / desired_bondlength| < cutoff, the constraint is called "fulfilled".

        Returns
        -------
        bool
            Is the constraint fulfilled?

        """

        # Get current length of the bond
        pos1 = structure.atoms.positions[self.iatom1]
        pos2 = structure.atoms.positions[self.iatom2]
        current_bondvector = pos2 - pos1
        current_bondlength = np.linalg.norm(current_bondvector)

        # Compare it to the desired bondlength
        if np.abs(1 - current_bondlength / self._desired_bondlength) < cutoff:
            return True
        else:
            return False

    def apply(self, structure):
        """Apply the constraint.

        Parameters
        ----------
        structure: riigid.Structure
            The constraint is applied to this supplied structure.

        """

        # Find the fragments corresponding to atoms 1 and 2
        for ifrag, frag in enumerate(structure.fragments):
            if self.iatom1 in frag.indices_in_structure:
                ifrag1 = ifrag
            if self.iatom2 in frag.indices_in_structure:
                ifrag2 = ifrag

        # Get the forces acting on each of the two fragments
        # f_frags = structure.get_forces_on_fragments(forces=forces)
        # f_frag1 = np.linalg.norm(f_frags[ifrag1])
        # f_frag2 = np.linalg.norm(f_frags[ifrag2])

        # Get current length and direction of the bond
        pos1 = structure.atoms.positions[self.iatom1]
        pos2 = structure.atoms.positions[self.iatom2]
        current_bondvector = pos2 - pos1
        current_bondlength = np.linalg.norm(current_bondvector)
        current_bondvector /= current_bondlength

        # Apply the constraint
        # Move fragments back together
        # Fragments with bigger force on them are moved less? or other way round?
        # weight1 = 1 - f_frag1 / (f_frag1 + f_frag2)
        # weight2 = 1 - weight1
        weight1 = 0.5
        weight2 = 0.5
        shift1 = (
            current_bondvector
            * (current_bondlength - self._desired_bondlength)
            * weight1
        )
        shift2 = (
            -current_bondvector
            * (current_bondlength - self._desired_bondlength)
            * weight2
        )
        structure.shift_and_rotate_a_fragment(
            fragment_index=ifrag1, shift=shift1, angle=0, axis=[1, 1, 1]
        )
        structure.shift_and_rotate_a_fragment(
            fragment_index=ifrag2, shift=shift2, angle=0, axis=[1, 1, 1]
        )
