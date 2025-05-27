from copy import copy, deepcopy

import numpy as np
from ase import Atoms

from riigid.library.rotation import (
    angle_between_vectors,
    rotmat,
    signed_angle_between_vectors,
)


class Fragment:
    """A collection of atoms with frozen bonds between them.

    In RIIGID a structure is a set of atoms separated into disjunctive subsets called fragments.
    The fragments are treated as rigid bodies, that is, the bonds between all atoms belonging to
    the same fragment are frozen.
    As already said, all these fragments together then form the structure.

    The orientation of a fragment can be defined using Euler angles and its position can be defined
    by its center of mass.

    Attributes
    ----------
    atoms: ase.atoms.Atoms
        The atoms forming the fragment.
    indices_in_structure: list of int
        Indices of the Fragment's atoms, relative to the Structure, that the Fragment is
        a part of.
    allowed_translation: str
        How shall the fragment be allowed to translate?
        See docstring of __init__ for more details.
    allowed_rotation: str
        Allows the user to set constraints on the rotation axis of a fragment.
        See docstring of __init__ for more details.
    body_fixed_axis_x/y/z: numpy.ndarray of shape (3,)
        The body-fixed axis system's xyz vectors (given in space-fixed coordinates)
    euler_angles: list of length 3
        The Euler angles of the fragment (alpha, beta, gamma); [°]
    inertia_matrix/_inv: numpy.ndarray of shape (3,3)
            The (inverse) inertia matrix of the fragment; [(Da*Å**2)]; (inverse:[1/(Da*Å**2)])

    """

    def __init__(
        self, atoms: Atoms, indices_in_structure, allowed_translation, allowed_rotation
    ):
        """Define a new fragment using an ASE Atoms object

        Parameters
        ----------
        atoms: ase.atoms.Atoms
            The atoms forming the fragment.
        indices_in_structure: list of int
            Indices of the Fragment's atoms, relative to the Structure, that the Fragment is
            a part of.
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

        """

        # Initialize using an already existing Atoms object
        self.atoms = deepcopy(atoms)
        self.indices_in_structure = indices_in_structure

        # Tranlation and rotation, which the fragment is allowed to do
        self.allowed_translation = allowed_translation
        self.allowed_rotation = allowed_rotation

        # Create body-fixed axis system, euler angles and inertia matrix
        self.body_fixed_axis_x = np.array([1.0, 0.0, 0.0])
        self.body_fixed_axis_y = np.array([0.0, 1.0, 0.0])
        self.body_fixed_axis_z = np.array([0.0, 0.0, 1.0])
        self.update_euler_angles()
        self.update_inertia_matrix()

    def update_body_fixed_axes(self, angle, axis):
        """Updates the body-fixed axis system.

        After each rotation of the fragment, the body-fixed axis system must be updated, in order
        to calculate the Euler angles of the fragment.

        Parameters
        ----------
        axis: list of length 3 or numpy.ndarray of shape (3,)
            The rotation axis
        angle: number
            The rotation angle; [°]

        """
        mat = rotmat(axis=axis, angle=angle)
        self.body_fixed_axis_x = mat @ self.body_fixed_axis_x
        self.body_fixed_axis_y = mat @ self.body_fixed_axis_y
        self.body_fixed_axis_z = mat @ self.body_fixed_axis_z

    def update_euler_angles(
        self,
        space_fixed_axis_x=[1, 0, 0],
        space_fixed_axis_y=[0, 1, 0],
        space_fixed_axis_z=[0, 0, 1],
    ):
        """Updates the Euler angles of the fragment.

        Via the orientation of a body-fixed axis system relative to a space-fixed axis system, the Euler
        angles of a fragment can be defined. After each update step, the body fixed axes change and the
        Euler angles must be updated.

        Convention: if z-axes of the two axis systems are parallel, set space_fixed_axis_x=N (line of nodes)

        (Body-fixed axes are given in space-fixed-coordinates.)

        See: https://en.wikipedia.org/wiki/Euler_angles

        Parameters
        ----------
        space_fixed_axis_x/y/z: list of length 3 or numpy.ndarray of shape (3,)
            The space-fixed axis system, relative to which the Euler angles are defined.
            Usually, the default values should be used.

        Returns
        -------
        list of length 3
            The three Euler angles; [°]

        """
        space_fixed_axis_x = np.array(space_fixed_axis_x)
        space_fixed_axis_y = np.array(space_fixed_axis_y)
        space_fixed_axis_z = np.array(space_fixed_axis_z)

        beta = angle_between_vectors(v1=space_fixed_axis_z, v2=self.body_fixed_axis_z)

        if beta == 0 or beta == 180:
            alpha = 0.0
            gamma = signed_angle_between_vectors(
                v1=space_fixed_axis_x,
                v2=self.body_fixed_axis_x,
                axis=self.body_fixed_axis_z,
            )
        else:
            N = np.cross(space_fixed_axis_z, self.body_fixed_axis_z)
            alpha = signed_angle_between_vectors(
                v1=space_fixed_axis_x, v2=N, axis=space_fixed_axis_z
            )
            gamma = signed_angle_between_vectors(
                v1=N, v2=self.body_fixed_axis_x, axis=self.body_fixed_axis_z
            )

        self.euler_angles = [alpha, beta, gamma]

        return copy(self.euler_angles)

    def update_inertia_matrix(self):
        """Get inertia matrix (and its inverse) of a fragment.

        The inertia matrix is defined relative to the fragment's center of mass.

        Note
        ----
        The inertia matrix must be updated after every rotation!

        Returns
        -------
        numpy.ndarray of shape (3,3)
            The inertia matrix of the fragment; [Da*Å**2]
        numpy.ndarray of shape (3,3)
            The inverse inertia matrix of the fragment; [1/(Da*Å**2)]

        """
        fragment_com = self.atoms.get_center_of_mass()
        inertia_matrix = np.zeros([3, 3])
        for j in range(3):
            for k in range(3):
                for atom in self.atoms:
                    r_l = atom.position - fragment_com
                    if j == k:
                        inertia_matrix[j, k] += atom.mass * (
                            np.linalg.norm(r_l) ** 2 - r_l[j] * r_l[k]
                        )
                    else:
                        inertia_matrix[j, k] += atom.mass * (-r_l[j] * r_l[k])
        if len(self.atoms) == 1:
            inertia_matrix_inv = np.zeros([3, 3])
        elif len(self.atoms) > 1:
            inertia_matrix_inv = np.linalg.inv(inertia_matrix)

        self.inertia_matrix = inertia_matrix
        self.inertia_matrix_inv = inertia_matrix_inv
        return copy(inertia_matrix), copy(inertia_matrix_inv)

    def update_rotation_properties(self, angle, axis):
        """Updates the body-fixed axis system, the Euler angles and the inertia matrix and its inverse.

        After each rotation of the fragment, these properties must be updated!

        Parameters
        ----------
        axis: list of length 3 or numpy.ndarray of shape (3,)
            The rotation axis
        angle: number
            The rotation angle; [°]

        """
        self.update_body_fixed_axes(angle=angle, axis=axis)
        self.update_euler_angles()
        self.update_inertia_matrix()

    def calculate_net_force_on_fragment(self, forces_structure):
        """Get the net force acting on the fragment.

        Parameters
        ----------
        forces_structure: numpy.ndarray of shape (n_atoms_structure, 3)
            Forces acting on the atoms in Structure that the fragment belongs to; [eV/Å]

        Returns
        -------
        numpy.ndarray of shape (3,)
            Allowed net force acting on the fragment; [eV/Å]
            This is calculated by removing parts from the full/raw net force,
            such that `self.allowed_translation` is fulfilled.
        numpy.ndarray of shape (3,)
            Raw/Full net force acting on the fragment; [eV/Å]

        """
        # Get forces on atoms that are part of the fragment
        forces = deepcopy(forces_structure[self.indices_in_structure, :])

        # Calculate raw net force
        raw_net_force_on_fragment = np.sum(forces, axis=0)

        # Calculate allowed net force
        allowed_net_force_on_fragment = deepcopy(raw_net_force_on_fragment)
        if "x" not in self.allowed_translation:
            allowed_net_force_on_fragment[0] = 0
        if "y" not in self.allowed_translation:
            allowed_net_force_on_fragment[1] = 0
        if "z" not in self.allowed_translation:
            allowed_net_force_on_fragment[2] = 0
        return allowed_net_force_on_fragment, raw_net_force_on_fragment

    def calculate_torque_on_fragment(self, forces_structure):
        """Get the net torque acting on the fragment (relative to its center of mass).

        Parameters
        ----------
        forces_structure: numpy.ndarray of shape (n_atoms_structure, 3)
            Forces acting on the atoms in Structure that the fragment belongs to; [eV/Å]

        Returns
        -------
        numpy.ndarray of shape (3,)
            Allowed torque acting on the fragment (relative to center of mass of fragment); [eV]
            This is calculated by removing parts from the full/raw torque,
            such that `self.allowed_rotation` is fulfilled.
        numpy.ndarray of shape (3,)
            Raw/Full torque acting on the fragment (relative to center of mass of fragment); [eV]

        """
        # Get forces on atoms that are part of the fragment
        forces = deepcopy(forces_structure[self.indices_in_structure, :])

        # Calculate raw torque
        fragment_com = self.atoms.get_center_of_mass()
        raw_torque_on_center = np.zeros(3)
        for i, atom in enumerate(self.atoms):
            r_i = atom.position
            r = fragment_com
            f_i = forces[i]
            raw_torque_on_center += np.cross(r_i - r, f_i)

        # Calculate the restricted torque as t_restricted = I @ [restrictions(I^-1 @ t)]
        rot_ax = self.inertia_matrix_inv @ raw_torque_on_center
        if "x" not in self.allowed_rotation:
            rot_ax[0] = 0
        if "y" not in self.allowed_rotation:
            rot_ax[1] = 0
        if "z" not in self.allowed_rotation:
            rot_ax[2] = 0
        allowed_torque_on_center = self.inertia_matrix @ rot_ax

        return allowed_torque_on_center, raw_torque_on_center

    def translate_by_shift(self, shift):
        """Translate fragment by simply shifting all atoms.

        Note
        ----
        DOES NOT enforce self.allowed_translations and self.allowed_rotations.

        Parameters
        ----------
        shift: numpy.ndarray of shape (3,) or equivalent list
            The vector to shift the fragment by; [Å]

        Returns
        -------
        numpy.ndarray of shape (3,)
            The translation vector; [Å]

        """
        if np.sum(np.abs(shift) ** 2) != 0:
            translation_vector = deepcopy(shift)
            translation_vector = np.array(translation_vector).reshape(-1)
            for atom in self.atoms:
                atom.position += translation_vector

        return translation_vector

    def translate_by_force(self, force_on_center, stepsize):
        """Translate fragment following the applied net force.

        Note
        ----
        DOES NOT enforce self.allowed_translations and self.allowed_rotations.

        Parameters
        ----------
        force_on_fragment: numpy.ndarray of shape (3,)
            The net force acting on the fragment; [eV/Å]
        stepsize: number
            Timestep; [Da*Å**2/eV]

        Returns
        -------
        numpy.ndarray of shape (3,)
            The translation vector; [Å]

        """
        fragment_mass = np.sum(self.atoms.get_masses())
        translation_vector = stepsize * force_on_center / fragment_mass

        for atom in self.atoms:
            atom.position += translation_vector

        return translation_vector

    def rotate_by_angle_and_axis(self, angle, axis):
        """Rotate fragment around its center of mass with given axis and angle.

        Note
        ----
        DOES NOT enforce self.allowed_translations and self.allowed_rotations.

        Parameters
        ----------
        axis: list of length 3 or numpy.ndarray of shape (3,)
            The rotation axis
        angle: number
            The rotation angle; [°]

        Returns
        -------
        numpy.ndarray of shape (3,)
            The rotation axis (normalized, if angle!=0);
        float
            The rotation angle; [°]

        """
        if len(self.atoms) > 1 and angle != 0:
            axis /= np.linalg.norm(axis)
            self.atoms.rotate(angle, axis, self.atoms.get_center_of_mass())
            self.update_rotation_properties(angle=angle, axis=axis)
            return axis, angle
        else:
            return np.array([1, 0, 0]), 0.0

    def rotate_by_torque(self, torque_on_center, stepsize):
        """Rotate fragment around its center of mass following the applied torque.

        Rotates the fragment and updates the rotation properties (Euler angles,
        body-fixed axes, inertia matrix) automatically.

        Note
        ----
        DOES NOT enforce self.allowed_translations and self.allowed_rotations.

        Parameters
        ----------
        torque_on_fragment: numpy.ndarray of shape (3,)
            Torque acting on the fragment (relative to center of mass of fragment); [eV]
        stepsize: number
            Timestep; [Da*Å**2/eV]

        Returns
        -------
        numpy.ndarray of shape (3,)
            The rotation axis (normalized, if angle!=0);
        float
            The rotation angle; [°]

        """
        if len(self.atoms) > 1:
            axis = self.inertia_matrix_inv @ torque_on_center
            angle = np.linalg.norm(axis) * (180 / np.pi) * stepsize  # in °

            if angle != 0:
                axis = axis / np.linalg.norm(axis)
                self.atoms.rotate(angle, axis, self.atoms.get_center_of_mass())
                self.update_rotation_properties(angle=angle, axis=axis)

            return axis, angle
        else:
            return np.array([1, 0, 0]), 0.0

    def rotate_by_euler_angles(self, alpha, beta, gamma):
        """Rotate fragment around its center of mass with given Euler angles to rotate by.

        Note
        ----
        DOES NOT enforce self.allowed_translations and self.allowed_rotations.

        Note
        ----
        This method rotates the fragment by alpha, beta, gamma, relative to current
        body-fixed axes! I.e., the final euler angles of the fragment, relative to the
        space-fixed axes (=self.euler_angles) will usually be different than alpha, beta
        and gamma. (Unless, self.euler_angles was [0,0,0] before calling this method.)

        Parameters
        ----------
        alpha, beta, gamma: float (0-360), float (0-180), float (0-360)
            The Euler angles; [°]

        Returns
        -------
        numpy.ndarray of shape (n_atoms_in_fragment,3)
            The positions of the fragment's atoms after the transformation; [Å]

        Raises
        ------
        ValueError
            If the given angles are not within the boundaries specified above.

        """
        if not (0 <= alpha <= 360):
            raise ValueError(
                f"Euler angle alpha {alpha} is not within the range [{0}, {360}]."
            )
        if not (0 <= beta <= 180):
            raise ValueError(
                f"Euler angle beta {beta} is not within the range [{0}, {180}]."
            )
        if not (0 <= gamma <= 360):
            raise ValueError(
                f"Euler angle gamma {gamma} is not within the range [{0}, {360}]."
            )

        self.rotate_by_angle_and_axis(angle=alpha, axis=self.body_fixed_axis_z)
        self.rotate_by_angle_and_axis(angle=beta, axis=self.body_fixed_axis_x)
        self.rotate_by_angle_and_axis(angle=gamma, axis=self.body_fixed_axis_z)
        return copy(self.atoms.positions)

    def move_by_forces(self, forces_structure, stepsize):
        """Rotate and translate the fragment.

        Rotates and translates the fragment and updates the rotation properties (Euler angles,
        body-fixed axes, inertia matrix) automatically

        Note
        ----
        DOES enforce self.allowed_translations and self.allowed_rotations.

        Parameters
        ----------
        forces_structure: numpy.ndarray of shape (n_atoms_structure, 3)
            Forces acting on the atoms in Structure that the fragment belongs to; [eV/Å]
        stepsize: number
            Timestep; [Da*Å**2/eV]

        Returns
        -------
        numpy.ndarray of shape (3,)
            The rotation axis (normalized, if angle!=0);
        float
            The rotation angle; [°]
        numpy.ndarray of shape (3,)
            The translation vector; [Å]

        """
        # Calculate force and torque acting on the fragment
        force_on_fragment, _ = self.calculate_net_force_on_fragment(
            forces_structure=forces_structure
        )
        torque_on_fragment, _ = self.calculate_torque_on_fragment(
            forces_structure=forces_structure
        )

        # Perform rotation and translation
        axis, angle = self.rotate_by_torque(
            torque_on_center=torque_on_fragment, stepsize=stepsize
        )
        shift = self.translate_by_force(
            force_on_center=force_on_fragment, stepsize=stepsize
        )

        return axis, angle, shift

    def move_random_step(self, displacement, angle, respect_restrictions, seed=1234):
        """Randomly rotate and translate the fragment.

        Useful to escape saddle points, especially when starting a new optimization.

        Parameters
        ----------
        displacement: number
            How far shall the fragment be translated; [Å]
        angle: number
            How much shall the fragment be rotated; [°]
        respect_restrictions: bool
            If True, self.allowed_translation/rotation is enforced.
            If False, rotation and translation in arbitrary directions is allowed temporarily.
            (After the random step, the restrictions are enforced again.)
        seed: int, default:1234
            The random seed used to generate the translation direction and rotation axis

        Returns
        -------
        numpy.ndarray of shape (3,)
            The rotation axis (normalized, if angle!=0);
        float
            The rotation angle; [°]
        numpy.ndarray of shape (3,)
            The translation vector; [Å]

        """
        # Prepare restrictions
        if not respect_restrictions:
            allowed_translation = "xyz"
            allowed_rotation = "xyz"
        else:
            allowed_translation = self.allowed_translation
            allowed_rotation = self.allowed_rotation

        # Get random translation direction and rotation axis
        backup_seed = np.random.randint(2**32 - 1)
        np.random.seed(seed)
        trans_dir = np.random.rand(3)
        rot_ax = np.random.rand(3)
        np.random.seed(backup_seed)

        # Apply restrictions
        if "x" not in allowed_translation:
            trans_dir[0] = 0
        if "y" not in allowed_translation:
            trans_dir[1] = 0
        if "z" not in allowed_translation:
            trans_dir[2] = 0
        if "x" not in allowed_rotation:
            rot_ax[0] = 0
        if "y" not in allowed_rotation:
            rot_ax[1] = 0
        if "z" not in allowed_rotation:
            rot_ax[2] = 0

        # Normalize translation direction and rotation axis
        if allowed_translation == "":
            trans_dir = np.array([1, 0, 0])
            displacement = 0.0
        else:
            trans_dir /= np.linalg.norm(trans_dir)
        if allowed_rotation == "":
            rot_ax = np.array([1, 0, 0])
            angle = 0.0
        else:
            rot_ax /= np.linalg.norm(rot_ax)

        # Apply translation
        self.translate_by_shift(shift=trans_dir * displacement)

        # Apply rotation
        self.rotate_by_angle_and_axis(angle=angle, axis=rot_ax)

        return rot_ax, angle, trans_dir * displacement

    #############################   TO BE DONE   ####################################################

    def set_to_euler_angles(self, alpha, beta, gamma):
        """
        tbd

        """
        raise Exception("Not fully implemented/tested yet.")
        """
        # First, remove current euler angles, s.t. body axes = space axes
        self.rotate_by_angle_and_axis(angle=-self.euler_angles[2], axis=self.body_fixed_axis_z)
        self.rotate_by_angle_and_axis(angle=-self.euler_angles[1], axis=self.body_fixed_axis_x)
        self.rotate_by_angle_and_axis(angle=-self.euler_angles[0], axis=self.body_fixed_axis_z)

        # Then, 
        self.rotate_by_angle_and_axis(angle=alpha, axis=self.body_fixed_axis_z)
        self.rotate_by_angle_and_axis(angle=beta, axis=self.body_fixed_axis_x)
        self.rotate_by_angle_and_axis(angle=gamma, axis=self.body_fixed_axis_z)
        """

    def apply_boundaries(self, xmin, xmax, ymin, ymax):
        """Needs to be fixed/redone from scratch"""
        com = self.atoms.get_center_of_mass()
        x = com[0]
        y = com[1]
        # print(fragment.get_center_of_mass())
        if x < xmin:
            self.atoms.positions[:, 0] = 2 * xmin - self.atoms.positions[:, 0]
        elif x > xmax:
            self.atoms.positions[:, 0] = 2 * xmax - self.atoms.positions[:, 0]

        if y < ymin:
            self.atoms.positions[:, 1] = 2 * ymin - self.atoms.positions[:, 1]
        elif y > ymax:
            self.atoms.positions[:, 1] = 2 * ymax - self.atoms.positions[:, 1]
        # print(fragment.get_center_of_mass())

        return self.atoms.positions.copy()
