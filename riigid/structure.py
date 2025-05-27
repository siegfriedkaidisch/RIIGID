from copy import deepcopy

import numpy as np

from riigid.fragment import Fragment
from riigid.library.misc import get_indices_of_atoms1_in_atoms2


class Structure:
    """Structure containing all the atoms and defined fragments.

    In RIIGID a structure is a set of atoms separated into disjunctive subsets called fragments.
    The fragments are treated as rigid bodies, that is, the bonds between all atoms belonging to
    the same fragment are frozen.
    As already said, all these fragments together then form the structure.

    Via the Structure class, fragments can be defined, forces and torques on all fragments can be
    calculated and the fragments can be moved accordingly.

    Attributes
    ----------
    atoms: ase.atoms.Atoms
        All atoms forming the structure, i.e, sum of all fragments.atoms (incl. rest_fragment).
    fragments: list of riigid.Fragment
        All fragments that have been defined.
    rest_fragment: riigid.Fragment
        All atoms that haven't been assigned a specific fragment together form the rest_fragment.

    """

    def __init__(self, atoms):
        """Initialize RIIGID structure object.

        Parameters
        ----------
        atoms: ase.atoms.Atoms
            To initialize a structure, an atoms object must be given, which can later be separated
            into fragments. It will also be possible to define a fragment by adding additional atoms later
            on (not yet implemented).

        """
        self.atoms = atoms
        self.fragments = []
        rest_fragment_indices, _ = get_indices_of_atoms1_in_atoms2(
            atoms1=self.atoms, atoms2=self.atoms
        )
        self.rest_fragment = Fragment(
            atoms=self.atoms,
            indices_in_structure=rest_fragment_indices,
            allowed_translation="",
            allowed_rotation="",
        )

    def define_fragment_by_indices(
        self, indices, allowed_translation, allowed_rotation
    ):
        """Define a RIIGID fragment by its indices.

        Define a new fragment by telling RIIGID the indices of the atoms (indices relative to
        Structure.atoms) that shall form the new fragment.

        Note
        ----
        Each atom has to belong to exactly one fragment! All atoms that the user never manually
        assigned to a specific fragment form another fragment together, called the rest_fragment.

        Parameters
        ----------
        indices: list of int
            The indices of the atoms forming the new fragment
        allowed_translation: str
            How shall the fragment be allowed to translate?
            If the string contains an "x", translation in x-direction is allowed, etc.
            E.g., to allow only translation in x- and y-direction, set allowed_translation="xy"
            To completely forbid any translation, use an empty string.
        allowed_rotation: str
            Allows the user to set constraints on the rotation axis of a fragment.
            Generally, the rotation axis (for a rigid body) is given my the matrix-vector product
            of the fragment's inverse inertia matrix with the torque acting on (the center of) the
            fragment. The rotation angle is then determined by the norm of the resulting vector.
            Using allowed_rotation, the user can apply the same logic as above to define, which components
            of the rotation axis shall be dropped.
            Examples:
            '' forbids any rotation
            'z' allows only rotation of the fragment around the (space-fixed) z-axis
            'xyz' allows for unrestricted rotation of the fragment

        Raises
        ------
        RuntimeError
            If an atom couldn't be found in the rest_fragment.

        """
        fragment_atoms = deepcopy(self.atoms[indices])
        rest_fragment_atoms = deepcopy(self.rest_fragment.atoms)

        new_fragment = Fragment(
            atoms=fragment_atoms,
            indices_in_structure=indices,
            allowed_translation=allowed_translation,
            allowed_rotation=allowed_rotation,
        )

        # Remove atoms of this new fragment from the rest-fragment
        (
            indices_of_new_fragment_in_rest_fragment,
            found,
        ) = get_indices_of_atoms1_in_atoms2(
            atoms1=fragment_atoms, atoms2=rest_fragment_atoms
        )
        if not found:
            raise RuntimeError(
                "Atoms not found in rest-fragment. Did you already include some of the atoms in another fragment? Every atom has to belong to exactly one fragment. (All atoms not assigned to any fragment form the rest-fragment.)"
            )
        del rest_fragment_atoms[indices_of_new_fragment_in_rest_fragment]

        # Update rest-fragment and append new fragment to list of fragments
        rest_fragment_indices, _ = get_indices_of_atoms1_in_atoms2(
            atoms1=rest_fragment_atoms, atoms2=self.atoms
        )
        self.rest_fragment = Fragment(
            atoms=rest_fragment_atoms,
            indices_in_structure=rest_fragment_indices,
            allowed_translation="",
            allowed_rotation="",
        )
        self.fragments.append(new_fragment)

    def define_fragment_by_adding_atoms(
        self, atoms, position, orientation, allowed_translation="", allowed_rotation=""
    ):
        """Define fragments by adding additional atoms to Structure.atoms.

        Not implemented yet!

        Things to consider:
        How to treat cells?
        How to position and orient the new fragment?
        If user wants to first use this fct and then use define_fragment_by_indices, shall user define
        indices relative to original Structure.atoms?

        """
        raise Exception("Not yet implemented")

    def update_atoms_attribute_from_fragments(self):
        """Update Structure.atoms after movement of fragments.

        When individual fragments are moved, the atoms object of the structure must also be updated
        accordingly.

        The function also makes sure that the order of the atoms in Structure.atoms
        is not changed!

        """
        for fragment in self.fragments + [self.rest_fragment]:
            self.atoms.positions[list(fragment.indices_in_structure), :] = deepcopy(
                fragment.atoms.positions
            )

    def calculate_energy_and_forces(self, calculator):
        """Calculate forces on all atoms and total energy.

        Parameters
        ----------
        calculator : ase.calculators.calculator.Calculator
            The used ASE calculator object

        Returns
        -------
        number
            The total energy; [eV]
        numpy.ndarray of shape (n_atoms, 3)
            Forces acting on the atoms in Structure.atoms; [eV/Å]

        """

        atoms = deepcopy(self.atoms)
        atoms.calc = calculator
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        return energy, forces

    def move(self, forces, stepsize):
        """Move the fragments according to the forces.

        Given the forces on all individual atoms and a stepsize, move the fragments.

        Note
        ----
        DOES enforce allowed_translations and allowed_rotations of fragments.

        Parameters
        ----------
        forces: numpy.ndarray of shape (n_atoms, 3)
            Forces acting on the atoms in Structure.atoms; [eV/Å]
        stepsize: number
            Timestep; [Da*Å**2/eV]

        Returns
        -------
        list of numpy.ndarray of shape (3,)
            The rotation axis (normalized, if angle!=0) of each fragment;
        list of float
            The rotation angle of each fragment; [°]
        list of numpy.ndarray of shape (3,)
            The translation vector of each fragment; [Å]

        """
        # Move the fragments
        axes = []
        angles = []
        shifts = []
        for i, fragment in enumerate(self.fragments):
            axis_i, angle_i, shift_i = fragment.move_by_forces(
                forces_structure=forces,
                stepsize=stepsize,
            )
            axes.append(axis_i)
            angles.append(angle_i)
            shifts.append(shift_i)

        # update self.atoms by summing up all fragments.atoms
        self.update_atoms_attribute_from_fragments()

        return axes, angles, shifts

    def move_random_step(self, displacement, angle, respect_restrictions, seed=1234):
        """Randomly rotate and translate the fragments.

        Useful to escape saddle points, especially when starting a new optimization.

        Parameters
        ----------
        displacement: number
            How far shall the fragments be translated; [Å]
        angle: number
            How much shall the fragments be rotated; [°]
        respect_restrictions: bool
            If True, fragment.allowed_translation/rotation is respected.
            If False, rotation and translation in arbitrary directions is allowed temporarily.
            (After the random step, the restrictions are respected again.)
        seed: int, default:1234
            The random seed used to generate the translation directions and rotation axes

        Returns
        -------
        list of numpy.ndarray of shape (3,)
            The rotation axis (normalized, if angle!=0) of each fragment;
        list of float
            The rotation angle of each fragment; [°]
        list of numpy.ndarray of shape (3,)
            The translation vector of each fragment; [Å]

        Note
        ----
        - The different fragments are rotated/translated around different axes/
        in different directions.
        - The rest_fragment is not moved!

        """

        # Generate one random seed per fragment, derived from the seed given as input to
        # this function.
        backup_seed = np.random.randint(2**32 - 1)
        np.random.seed(seed)
        seeds_for_fragments = [
            np.random.randint(2**32 - 1) for _ in range(len(self.fragments))
        ]
        np.random.seed(backup_seed)

        # Randomly move all fragments except the rest_fragment
        axes = []
        angles = []
        shifts = []
        for i, fragment in enumerate(self.fragments):
            axis_i, angle_i, shift_i = fragment.move_random_step(
                displacement=displacement,
                angle=angle,
                respect_restrictions=respect_restrictions,
                seed=seeds_for_fragments[i],
            )
            axes.append(axis_i)
            angles.append(angle_i)
            shifts.append(shift_i)

        # update self.atoms by summing up all fragments.atoms
        self.update_atoms_attribute_from_fragments()

        return axes, angles, shifts

    def shift_and_rotate_a_fragment(self, fragment_index, shift, angle, axis):
        """Shift and rotate a fragment from Structure.fragments.

        Can be useful for optimizers, e.g. GPR.

        Note
        ----
        DOES NOT enforce allowed_translations and allowed_rotations of fragments.

        Parameters
        ----------
        fragment_index: int
            Structure.fragments[fragment_index] will be shifted and rotated
        shift: numpy.ndarray of shape (3,) or equivalent list
            The vector to shift the fragment by; [Å]
        angle: number
            How much shall the fragments be rotated; [°]
        axis: list of length 3 or numpy.ndarray of shape (3,)
            The rotation axis

        Returns
        -------
        numpy.ndarray of shape (n_atoms_in_structure,3)
            The positions of the structure's atoms after the transformation; [Å]

        """
        # Select the fragment and rotate/translate it
        frag = self.fragments[fragment_index]
        frag.translate_by_shift(shift=shift)
        frag.rotate_by_angle_and_axis(angle=angle, axis=axis)

        # Update Structure attributes and return new atomic positions
        self.update_atoms_attribute_from_fragments()
        new_positions = deepcopy(self.atoms.positions)
        return new_positions
