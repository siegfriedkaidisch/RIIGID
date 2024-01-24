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

    def __init__(self, atoms: Atoms, allowed_translation, allowed_rotation):
        """Define a new fragment using an ASE Atoms object

        Parameters
        ----------
        atoms: ase.atoms.Atoms
            The atoms forming the fragment.
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

    def calculate_net_force_on_fragment(self, forces):
        """Get the net force acting on the fragment.

        Parameters
        ----------
        forces: numpy.ndarray of shape (n_atoms_in_fragment, 3)
            Forces acting on the atoms in the fragment; [eV/Å]

        Returns
        -------
        numpy.ndarray of shape (3,)
            Net force acting on the fragment; [eV/Å]

        """
        net_force_on_fragment = np.sum(forces, axis=0)
        return net_force_on_fragment

    def calculate_torque_on_fragment(self, forces):
        """Get the net torque acting on the fragment (relative to its center of mass).

        Parameters
        ----------
        forces: numpy.ndarray of shape (n_atoms_in_fragment, 3)
            Forces acting on the atoms in the fragment; [eV/Å]

        Returns
        -------
        numpy.ndarray of shape (3,)
            Torque acting on the fragment (relative to center of mass of fragment); [eV]

        """
        fragment_com = self.atoms.get_center_of_mass()
        torque_on_center = np.zeros(3)
        for i, atom in enumerate(self.atoms):
            r_i = atom.position
            r = fragment_com
            f_i = forces[i]
            torque_on_center += np.cross(r_i - r, f_i)
        return torque_on_center

    def move(self, force_on_fragment, torque_on_fragment, stepsize):
        """Rotate and translate the fragment.

        Rotates and translates the fragment and updates the rotation properties (Euler angles,
        body-fixed axes, inertia matrix) automatically

        Parameters
        ----------
        force_on_fragment: numpy.ndarray of shape (3,)
            The net force acting on the fragment; [eV/Å]
        torque_on_fragment: numpy.ndarray of shape (3,)
            Torque acting on the fragment (relative to center of mass of fragment); [eV]
        stepsize: number
            Timestep; [Da*Å**2/eV]

        """
        self.rotate_by_torque(torque_on_center=torque_on_fragment, stepsize=stepsize)
        self.translate_by_force(force_on_center=force_on_fragment, stepsize=stepsize)

    def rotate_by_torque(self, torque_on_center, stepsize):
        """Rotate fragment around its center of mass following the applied torque.

        Rotates the fragment and updates the rotation properties (Euler angles,
        body-fixed axes, inertia matrix) automatically.

        Parameters
        ----------
        torque_on_fragment: numpy.ndarray of shape (3,)
            Torque acting on the fragment (relative to center of mass of fragment); [eV]
        stepsize: number
            Timestep; [Da*Å**2/eV]

        Returns
        -------
        numpy.ndarray of shape (n_atoms_in_fragment,3)
            The positions of the fragment's atoms after the transformation; [Å]

        """
        axis, angle = self.get_rotation_axis_and_angle_from_torque(
            torque_on_center=torque_on_center, stepsize=stepsize
        )

        if angle != 0:
            self.atoms.rotate(angle, axis, self.atoms.get_center_of_mass())
            self.update_rotation_properties(angle=angle, axis=axis)

        return copy(self.atoms.positions)

    def get_rotation_axis_and_angle_from_torque(self, torque_on_center, stepsize=1.0):
        """Get the (normalized) rotation axis and angle corresponding to the applied applied torque and the given stepsize.

        Parameters
        ----------
        torque_on_fragment: numpy.ndarray of shape (3,)
            Torque acting on the fragment (relative to center of mass of fragment); [eV]
        stepsize: number, default:1.0
            Timestep; [Da*Å**2/eV]

        Note
        ----
        The restrictions on the movement (Fragment.allowed_rotation) ARE taken into account!

        Returns
        -------
        numpy.ndarray of shape (3,)
            The normalized rotation axis
        number
            The rotation angle; [°]

        """
        axis = self.inertia_matrix_inv @ torque_on_center
        if "x" not in self.allowed_rotation:
            axis[0] = 0
        if "y" not in self.allowed_rotation:
            axis[1] = 0
        if "z" not in self.allowed_rotation:
            axis[2] = 0

        angle = np.linalg.norm(axis) * (180 / np.pi) * stepsize  # in °
        if angle != 0:
            axis = axis / np.linalg.norm(axis)

        return axis, angle

    def translate_by_force(self, force_on_center, stepsize):
        """Translate fragment following the applied net force.

        Parameters
        ----------
        force_on_fragment: numpy.ndarray of shape (3,)
            The net force acting on the fragment; [eV/Å]
        stepsize: number
            Timestep; [Da*Å**2/eV]

        Returns
        -------
        numpy.ndarray of shape (n_atoms_in_fragment,3)
            The positions of the fragment's atoms after the transformation; [Å]

        """
        translation_vector = self.get_translation_vector_from_force(
            force_on_center=force_on_center, stepsize=stepsize
        )
        for atom in self.atoms:
            atom.position += translation_vector
        # self.update_rotation_properties(angle=0.0, axis=[0.0,0.0,1.0]) #rotation properties are unaffected by translations -> not needed

        return copy(self.atoms.positions)

    def get_translation_vector_from_force(self, force_on_center, stepsize=1.0):
        """Get the fragment's translation vector corresponding to the net force and the stepsize.

        Parameters
        ----------
        force_on_fragment: numpy.ndarray of shape (3,)
            The net force acting on the fragment; [eV/Å]
        stepsize: number, default:1.0
            Timestep; [Da*Å**2/eV]

        Note
        ----
        The restrictions on the movement (Fragment.allowed_translation) ARE taken into account!

        Returns
        -------
        numpy.ndarray of shape (3,)
            The translation vector; [Å]

        """
        fragment_mass = np.sum(self.atoms.get_masses())
        translation_vector = np.zeros(3)
        if "x" in self.allowed_translation:
            translation_vector[0] = stepsize * force_on_center[0] / fragment_mass
        if "y" in self.allowed_translation:
            translation_vector[1] = stepsize * force_on_center[1] / fragment_mass
        if "z" in self.allowed_translation:
            translation_vector[2] = stepsize * force_on_center[2] / fragment_mass

        return translation_vector

    def rotate_by_angle_and_axis(self, angle, axis):
        """Rotate fragment around its center of mass with given axis and angle.

        WARNING: this functions currently ignores self.allowed_rotations!

        Parameters
        ----------
        axis: list of length 3 or numpy.ndarray of shape (3,)
            The rotation axis
        angle: number
            The rotation angle; [°]

        Returns
        -------
        numpy.ndarray of shape (n_atoms_in_fragment,3)
            The positions of the fragment's atoms after the transformation; [Å]

        """
        self.atoms.rotate(angle, axis, self.atoms.get_center_of_mass())
        self.update_rotation_properties(angle=angle, axis=axis)
        return copy(self.atoms.positions)

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
            If True, self.allowed_translation/rotation is respected.
            If False, rotation and translation in arbitrary directions is allowed temporarily.
            (After the random step, the restrictions are respected again.)
        seed: int, default:1234
            The random seed used to generate the translation direction and rotation axis

        Returns
        -------
        numpy.ndarray of shape (n_atoms_in_fragment,3)
            The positions of the fragment's atoms after the transformation; [Å]

        """
        backup_allowed_translation = copy(self.allowed_translation)
        backup_allowed_rotation = copy(self.allowed_rotation)
        if not respect_restrictions:
            self.allowed_translation = "xyz"
            self.allowed_rotation = "xyz"

        # Set like this, s.t. rotation angle and displacement is as requested
        stepsize = 1.0
        inertia_matrix_inv = np.eye(3)
        fragment_mass = np.sum(self.atoms.get_masses())

        backup_seed = np.random.randint(2**32 - 1)
        np.random.seed(seed)
        force = np.random.rand(3)
        torque = np.random.rand(3)
        np.random.seed(backup_seed)

        # Set like this, s.t. rotation angle and displacement is as requested
        force /= np.linalg.norm(force)
        torque /= np.linalg.norm(torque)
        force *= displacement * fragment_mass
        torque *= angle * (np.pi / 180)

        # Translate fragment
        self.translate_by_force(force_on_center=force, stepsize=stepsize)
        # Rotate fragment
        self.inertia_matrix_inv = inertia_matrix_inv  # temporarily set to diag, such that fragment is rotated arbitrarily
        self.rotate_by_torque(torque_on_center=torque, stepsize=stepsize)

        self.allowed_translation = copy(backup_allowed_translation)
        self.allowed_rotation = copy(backup_allowed_rotation)
        return copy(self.atoms.positions)

    #########################################################################################################

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
