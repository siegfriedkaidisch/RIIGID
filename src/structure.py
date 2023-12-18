from copy import deepcopy
import numpy as np

from fragment import Fragment
from misc_functions import get_indices_of_atoms1_in_atoms2


class Structure:
    """Structure containing all the atoms and defined fragments.

    In RIGID a structure is a set of atoms separated into disjunctive subsets called fragments.
    The fragments are treated as rigid bodies, that is, the bonds between all atoms belonging to
    the same fragment are frozen.
    As already said, all these fragments together then form the structure.

    Via the Structure class, fragments can be defined, forces and torques on all fragments can be
    calculated and the fragments can be moved accordingly.

    Attributes
    ----------
    

    """

    def __init__(self, atoms):
        """Initialize RIGID structure object.

        Parameters
        ----------
        atoms: ase.atoms.Atoms
            To initialize a structure, an atoms object must be given, which can later be separated
            into fragments. It will also be possible to define a fragment by adding additional atoms later
            on (not yet implemented).

        """
        self.atoms = atoms
        self.fragments = []
        self.rest_fragment = Fragment(
            atoms=self.atoms, allowed_translation="", allowed_rotation=""
        )

    def define_fragment_by_indices(
        self, indices, allowed_translation, allowed_rotation
    ):
        """Define a RIGID fragment by its indices.

        Define a new fragment by telling RIGID the indices of the atoms (indices relative to
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
            allowed_translation=allowed_translation,
            allowed_rotation=allowed_rotation,
        )

        # Remove atoms of this new fragment from the rest-fragment
        (
            indices_of_new_fragment_in_rest_fragment,
            found,
        ) = get_indices_of_atoms1_in_atoms2(
            atoms1=fragment_atoms, atoms2=fragment_atoms
        )
        if not found:
            raise RuntimeError(
                "Atoms not found in rest-fragment. Did you already include some of the atoms in another fragment? Every atom has to belong to exactly one fragment. (All atoms not assigned to any fragment form the rest-fragment.)"
            )
        del rest_fragment_atoms[indices_of_new_fragment_in_rest_fragment]

        # Update rest-fragment and append new fragment to list of fragments
        self.rest_fragment = Fragment(
            atoms=rest_fragment_atoms, allowed_translation="", allowed_rotation=""
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
        print("Not yet implemented")

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
            Forces acting on the atoms in Structure.atoms; [eV/AA]

        """

        atoms = deepcopy(self.atoms)
        atoms.set_calculator(calculator)
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        return energy, forces

    def get_indices_of_fragments(self):
        """Get the indices (relative to Structure.atoms) of the atoms in each fragment.

        Returns
        -------
        list of lists of int
            A list containing one list per fragment (excluding the rest_fragment) containing
            the indices of the fragment

        Raises
        ------
        RuntimeError
            Raises an exception if not all indices of the fragment's atoms could be found.

        """
        fragments_indices = []
        for fragment in self.fragments:
            fragment_indices, found = get_indices_of_atoms1_in_atoms2(
                atoms1=fragment.atoms, atoms2=self.atoms, cutoff=1e-4
            )
            if not found:
                raise RuntimeError("Error while looking for indices of fragment...")
            fragments_indices.append(fragment_indices)
        return fragments_indices

    def get_forces_on_fragments(self, forces):
        """Assign forces to fragments.

        Given the forces on all individual atoms, get one numpy array per fragment, containing
        the forces on all the atoms inside the fragment.

        Parameters
        ----------
        forces: numpy.ndarray of shape (n_atoms, 3)
            Forces acting on the atoms in Structure.atoms; [eV/AA]

        Returns
        -------
        list of numpy.ndarrays of shape (depends_on_fragment, 3)
            One numpy array per fragment with the forces on atoms belonging to the fragment

        """
        fragments_indices = self.get_indices_of_fragments()
        forces_on_fragments = [forces[indices_i] for indices_i in fragments_indices]
        return forces_on_fragments

    def calculate_net_force_on_fragments(self, forces):
        """Get net force on each fragment.

        Given the forces on all individual atoms, get the net force acting on each fragment.

        Parameters
        ----------
        forces: numpy.ndarray of shape (n_atoms, 3)
            Forces acting on the atoms in Structure.atoms; [eV/AA]

        Returns
        -------
        list of numpy.ndarrays of shape (3,)
            Net force on each fragment; [eV/AA]

        """
        forces_on_fragments = self.get_forces_on_fragments(forces=forces)
        net_force_on_fragments = []
        for i, fragment in enumerate(self.fragments):
            net_force_on_fragment = fragment.calculate_net_force_on_fragment(
                forces=forces_on_fragments[i]
            )
            net_force_on_fragments.append(net_force_on_fragment)
        return net_force_on_fragments

    def calculate_torque_on_fragments(self, forces):
        """Get torque on each fragment, relative to the fragment's center of mass.

        Given the forces on all individual atoms, get the net torque acting on each fragment.
        (The torque is calculated relative to the center of mass of the fragment.)

        Parameters
        ----------
        forces: numpy.ndarray of shape (n_atoms, 3)
            Forces acting on the atoms in Structure.atoms; [eV/AA]

        Returns
        -------
        list of numpy.ndarrays of shape (3,)
            Torque on each fragment; [eV]

        """
        forces_on_fragments = self.get_forces_on_fragments(forces=forces)
        torque_on_fragments = []
        for i, fragment in enumerate(self.fragments):
            torque_on_fragment = fragment.calculate_torque_on_fragment(
                forces=forces_on_fragments[i]
            )
            torque_on_fragments.append(torque_on_fragment)
        return torque_on_fragments

    def update_atoms_attribute_from_fragments(self):
        """Update Structure.atoms after movement of fragments.

        When individual fragments are moved, the atoms object of the structure must also be updated
        accordingly.

        """
        self.atoms = self.rest_fragment.atoms + sum(
            [fragment.atoms for fragment in self.fragments]
        )

    def move(self, forces, stepsize):
        """Move the fragments according to the forces.

        Given the forces on all individual atoms and a stepsize, move the fragments.

        The functions first calculates the net force and the torque acting on each fragment.
        Then, the fragments are moved.

        Parameters
        ----------
        forces: numpy.ndarray of shape (n_atoms, 3)
            Forces acting on the atoms in Structure.atoms; [eV/AA]
        stepsize: number
            Timestep; [Dalton*AA**2/eV]

        Returns
        -------
        number
            The farthest distance an atom was moved in this update step; [AA]
        numpy.ndarray of shape (n_atoms, 3)
            xyz displacement of each atom; [AA]

        """
        old_positions = deepcopy(self.atoms.positions)

        force_on_fragments = self.calculate_force_on_fragments(forces=forces)
        torque_on_fragments = self.calculate_torque_on_fragments(forces=forces)

        for i, fragment in enumerate(self.fragments):
            fragment.move(
                force_on_fragment=force_on_fragments[i],
                torque_on_fragment=torque_on_fragments[i],
                stepsize=stepsize,
            )

        # update self.atoms by summing up all fragments.atoms
        self.update_atoms_attribute_from_fragments()

        new_positions = deepcopy(self.atoms.positions)

        atomic_displacements = new_positions - old_positions
        max_atomic_displacement = np.max(np.linalg.norm(atomic_displacements, axis=1))
        return max_atomic_displacement, atomic_displacements
