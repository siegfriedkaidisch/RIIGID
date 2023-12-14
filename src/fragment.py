import numpy as np
from copy import deepcopy, copy

from ase import Atoms

from rotation_functions import angle_between_vectors, angle_between_vectors2, rotmat
from misc_functions import get_indices_of_atoms1_in_atoms2

class Fragment():
    """
    The user is advised to not directly modify Fragment's properties, in order not to
    mess up rotation properties.
    """
    def __init__(self, atoms:Atoms, allowed_translation, allowed_rotation):
        # Initialize using an already existing Atoms object
        self.atoms = deepcopy(atoms)

        # Tranlation and rotation, which the fragment is allowed to do 
        self.allowed_translation = allowed_translation
        self.allowed_rotation = allowed_rotation

        # Create body-fixed axis system, euler angles and inertia matrix
        self.body_fixed_axis_x = np.array([1.0,0.0,0.0])
        self.body_fixed_axis_y = np.array([0.0,1.0,0.0])
        self.body_fixed_axis_z = np.array([0.0,0.0,1.0])
        self.update_euler_angles()
        self.update_inertia_matrix()

        return None
    
    def update_body_fixed_axes(self, angle, axis):
        """
        Updates the body-fixed axis system
        """
        mat = rotmat(axis=axis,angle=angle)
        self.body_fixed_axis_x = mat@self.body_fixed_axis_x
        self.body_fixed_axis_y = mat@self.body_fixed_axis_y
        self.body_fixed_axis_z = mat@self.body_fixed_axis_z
        return None

    def update_euler_angles(self, space_fixed_axis_x=[1,0,0],space_fixed_axis_y=[0,1,0],space_fixed_axis_z=[0,0,1]):
        """
        body-fixed vs space-fixed axis system (body-axes given in space-coords)
        convention: if z axes are parallel, set space_fixed_axis_x=N
        """
        space_fixed_axis_x = np.array(space_fixed_axis_x)
        space_fixed_axis_y = np.array(space_fixed_axis_y)
        space_fixed_axis_z = np.array(space_fixed_axis_z)
        
        beta  = angle_between_vectors(v1=space_fixed_axis_z, v2=self.body_fixed_axis_z)
        
        if beta == 0 or beta==180:
            alpha = 0.0
            gamma = angle_between_vectors2(v1=space_fixed_axis_x,v2=self.body_fixed_axis_x, axis=self.body_fixed_axis_z)
        else:
            N = np.cross(space_fixed_axis_z, self.body_fixed_axis_z)
            alpha = angle_between_vectors2(v1=space_fixed_axis_x,v2=N, axis=space_fixed_axis_z)
            gamma = angle_between_vectors2(v1=N,v2=self.body_fixed_axis_x, axis=self.body_fixed_axis_z)
        
        self.euler_angles = [alpha, beta, gamma]

        return copy(self.euler_angles)

    def update_inertia_matrix(self):
        """
        Get inertia matrix (and its inverse) of a fragment.

        Inputs:
            fragment: ase.atoms.Atoms
                The fragment whose inertia matrix (and inverse) will be calculated

        Returns:
            np.ndarray of shape (3,3)
                The inertia matrix of the fragment (in Dalton*Angstroem**2)
            np.ndarray of shape (3,3)
                The inverse inertia matrix of the fragment (in 1/(Dalton*Angstroem**2))
        """
        fragment_com = self.atoms.get_center_of_mass()
        inertia_matrix = np.zeros([3,3])
        for j in range(3):
            for k in range(3):
                for atom in self.atoms:
                    r_l = atom.position - fragment_com
                    if j==k:
                        inertia_matrix[j,k] += atom.mass * ( np.linalg.norm(r_l)**2 - r_l[j]*r_l[k] ) 
                    else:
                        inertia_matrix[j,k] += atom.mass * (                - r_l[j]*r_l[k] ) 
        inertia_matrix_inv = np.linalg.inv(inertia_matrix)

        self.inertia_matrix = inertia_matrix
        self.inertia_matrix_inv = inertia_matrix_inv
        return copy(inertia_matrix), copy(inertia_matrix_inv)
    
    def update_rotation_properties(self, angle, axis):
        self.update_body_fixed_axes(angle=angle, axis=axis)
        self.update_euler_angles()
        self.update_inertia_matrix()
        return None
    
    def calculate_net_force_on_fragment(self, forces):
        """
        Get the net force acting on the fragment.

        Inputs:
            fragment_indices: list of length n_atoms_in_fragment
                List containing indices of the fragment's atoms in "full"
            f: np.ndarray of shape (n_atoms_in_full_system, 3)
                Forces acting on the atoms in "full" (in eV/Angstroem)

        Returns:
            np.ndarray of shape (3,)
                Net force acting on the fragment (in eV/Angstroem)
        """
        net_force_on_fragment = np.sum(forces, axis=0)
        return net_force_on_fragment

    def calculate_torque_on_fragment(self, forces):
        """
        Get the net torque acting on the fragment (relative to its center of mass).

        Inputs:
            full: ase.atoms.Atoms
                The full system (surface+fragment) under study
            fragment_indices: list of length n_atoms_in_fragment
                List containing indices of the fragment's atoms in "full"
            f: np.ndarray of shape (n_atoms_in_full_system, 3)
                Forces acting on the atoms in "full" (in eV/Angstroem)

        Returns:
            np.ndarray of shape (3,)
                Net torque acting on the fragment (relative to center of mass of fragment) (in eV)
        """
        fragment_com = self.atoms.get_center_of_mass()
        for i,atom in enumerate(self.atoms):
            r_i = atom.position
            r   = fragment_com
            f_i = forces[i]
            torque_on_center += np.cross(r_i-r, f_i)
        return torque_on_center

    def move(self, force_on_fragment, torque_on_fragment, stepsize):
        self.rotate(torque_on_center=torque_on_fragment, stepsize=stepsize)
        self.translate(force_on_center=force_on_fragment, stepsize=stepsize)

    def rotate(self, torque_on_center, stepsize):
        """
        Rotate fragment around its center of mass following the applied torque

        Inputs:
            fragment: ase.atoms.Atoms
                The fragment to be rotated
            fragment_inertia_inv: np.ndarray of shape (3,3)
                The inverse inertia matrix of the fragment (in 1/(Dalton*Angstroem**2))
            t_center: np.ndarray of shape (3,) or list of length 3
                Net torque acting on the fragment (relative to center of mass of fragment) (in eV)
            stepsize_rot: number
                Timestep (in Dalton*Angstroem**2/eV)
            z_only_rot: Bool
                Rotate only around the z-axis?

        Returns:
            np.ndarray of shape (n_atoms_in_fragment,3)
                The positions (in Angstroem) of the fragment's atoms after the transformation
        """
        tmp = self.inertia_matrix_inv@torque_on_center

        if self.allowed_rotation == "xyz":
            angle = np.linalg.norm(tmp) * (180/np.pi) * stepsize  # in degrees
            if angle != 0:
                axis = tmp/np.linalg.norm(tmp)
                self.atoms.rotate(angle,axis,self.atoms.get_center_of_mass())
                self.update_rotation_properties(angle=angle, axis=axis)
        elif self.allowed_rotation == 'z':
            angle = tmp[2] * (180/np.pi) * stepsize
            if angle != 0:
                axis = np.array([0.0,0.0,1.0])
                self.atoms.rotate(angle,axis,self.atoms.get_center_of_mass())
                self.update_rotation_properties(angle=angle, axis=axis)
        else:
            raise Exception('Input for allowed rotation of fragment not supported: '+str(self.allowed_rotation))

        return copy(self.atoms.positions)

    def translate(self, force_on_center, stepsize):
        """
        Translate fragment in direction of the applied force

        Inputs:
            fragment: ase.atoms.Atoms
                The fragment to be translated
            f_center: np.ndarray of shape (3,) or list of length 3
                Net force acting on the fragment (in eV/Angstroem)
            stepsize_trans_x/y/z: number
                Timesteps; usually all three should have the same value; (in Dalton*Angstroem**2/eV)

        Returns:
            np.ndarray of shape (n_atoms_in_fragment,3)
                The positions (in Angstroem) of the fragment's atoms after the transformation
        """
        fragment_mass = np.sum(self.atoms.get_masses())
        if self.allowed_translation == 'xyz':
            for atom in self.atoms:
                atom.position[0] += stepsize * force_on_center[0]/fragment_mass
                atom.position[1] += stepsize * force_on_center[1]/fragment_mass
                atom.position[2] += stepsize * force_on_center[2]/fragment_mass
            #self.update_rotation_properties(angle=0.0, axis=[0.0,0.0,1.0]) #rotation properties are unaffected by translations -> not needed
        else:
            raise Exception('Input for allowed translation of fragment not supported: '+str(self.allowed_rotation))

        return copy(self.atoms.positions)
    




















    #########################################################################################
    

    def rotate2(self, angle, axis):
        """
        Rotate fragment around com and rotate its axis system and update euler angle
        Currently ignores self.allowed_rotations! Change?
        """   
        self.atoms.rotate(angle, axis, self.atoms.get_center_of_mass())
        self.update_rotation_properties(angle=angle, axis=axis)
        return copy(self.atoms.positions)

    def move_random_step(self, displacement, angle, seed):
        """
        Needs to be fixed
        """
        stepsize = 1.0
        inertia_matrix_inv  = np.eye(3)
        fragment_mass = np.sum(self.atoms.get_masses())
        
        backup_seed = np.random.randint(2**32 - 1)
        np.random.seed(seed)
        force  = np.random.rand(3)
        torque = np.random.rand(3) 
        np.random.seed(backup_seed)
        
        force  /= np.linalg.norm(force)
        torque /= np.linalg.norm(torque)
        force  *= displacement * fragment_mass
        torque *= angle * (np.pi/180)
        
        # Translate fragment
        self.translate(force_on_center=force, stepsize=stepsize)
        # Rotate fragment 
        self.inertia_matrix_inv = inertia_matrix_inv #temporarily set to diag, such that fragment is rotated arbitrarily
        self.rotate(torque_on_center=torque, stepsize=stepsize)
        return copy(self.atoms.positions)
    
    def apply_boundaries(self, xmin, xmax, ymin, ymax):
        """
        Needs to be fixed
        """
        com = self.atoms.get_center_of_mass()
        x = com[0]
        y = com[1]
        #print(fragment.get_center_of_mass())
        if x < xmin:
            self.atoms.positions[:,0] = 2* xmin - self.atoms.positions[:,0]
        elif x > xmax:
            self.atoms.positions[:,0] = 2* xmax - self.atoms.positions[:,0]

        if y < ymin:
            self.atoms.positions[:,1] = 2* ymin - self.atoms.positions[:,1]
        elif y > ymax:
            self.atoms.positions[:,1] = 2* ymax - self.atoms.positions[:,1]
        #print(fragment.get_center_of_mass())

        return self.atoms.positions.copy()