#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fr Nov 24 2023

Python Class for Rigid Atoms

@author: kaidisch_siegfried
"""
import numpy as np
from copy import deepcopy, copy

from ase import Atoms

from lib.rotation_functions import angle_between_vectors, angle_between_vectors2, rotmat

#Problem: if user directly interacts with Fragment.atoms, e.g. by rotating it, the rotation properties
# of the Fragment are not updated...
class Fragment():
    '''
    The user is advised to not directly modify Fragment.atoms!
    '''
    def __init__(self, atoms:Atoms):
        # Initialize using an already existing Atoms object
        self.atoms = deepcopy(atoms)

        # Create body-fixed axis system, euler angles and inertia matrix
        self.body_fixed_axis_x = np.array([1.0,0.0,0.0])
        self.body_fixed_axis_y = np.array([0.0,1.0,0.0])
        self.body_fixed_axis_z = np.array([0.0,0.0,1.0])
        self.update_euler_angles()
        self.update_inertia_matrix()

        return None
    
    def update_body_fixed_axes(self, angle, axis):
        '''
        Updates the body-fixed axis system
        '''
        mat = rotmat(axis=axis,angle=angle)
        self.body_fixed_axis_x = mat@self.body_fixed_axis_x
        self.body_fixed_axis_y = mat@self.body_fixed_axis_y
        self.body_fixed_axis_z = mat@self.body_fixed_axis_z
        return None

    def update_euler_angles(self, space_fixed_axis_x=[1,0,0],space_fixed_axis_y=[0,1,0],space_fixed_axis_z=[0,0,1]):
        '''
        body-fixed vs space-fixed axis system (body-axes given in space-coords)
        convention: if z axes are parallel, set space_fixed_axis_x=N
        '''
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
        '''
        Get inertia tensor (and its inverse) of a fragment.

        Inputs:
            fragment: ase.atoms.Atoms
                The fragment whose inertia matrix (and inverse) will be calculated

        Returns:
            np.ndarray of shape (3,3)
                The inertia matrix of the fragment (in Dalton*Angstroem**2)
            np.ndarray of shape (3,3)
                The inverse inertia matrix of the fragment (in 1/(Dalton*Angstroem**2))
        '''
        fragment_com = self.atoms.get_center_of_mass()
        fragment_inertia = np.zeros([3,3])
        for j in range(3):
            for k in range(3):
                for atom in self.atoms:
                    r_l = atom.position - fragment_com
                    if j==k:
                        fragment_inertia[j,k] += atom.mass * ( np.linalg.norm(r_l)**2 - r_l[j]*r_l[k] ) 
                    else:
                        fragment_inertia[j,k] += atom.mass * (                - r_l[j]*r_l[k] ) 
        fragment_inertia_inv = np.linalg.inv(fragment_inertia)

        self.fragment_inertia = fragment_inertia
        self.fragment_inertia_inv = fragment_inertia_inv
        return copy(fragment_inertia), copy(fragment_inertia_inv)
    
    def update_rotation_properties(self, angle, axis):
        self.update_body_fixed_axes(angle=angle, axis=axis)
        self.update_euler_angles()
        self.update_inertia_matrix()
        return None
    
    def get_indices(self, full, cutoff=1e-4):
        '''
        Find the indices of the Fragment's atoms in the full system 
        '''
        fragment_indices = []
        for fragment_atom in self.atoms:
            for full_atom in full:
                if np.linalg.norm(fragment_atom.position - full_atom.position) < cutoff:
                    fragment_indices.append(full_atom.index)
        return copy(fragment_indices)
    
    def get_torque_on_center(self, full, fragment_indices, f):
        '''
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
        '''
        fragment_com = self.atoms.get_center_of_mass()
        t_center = np.array([0.0,0.0,0.0])
        for index in fragment_indices:
            r_i = full[index].position
            r   = fragment_com
            f_i = f[index]
            t_center += np.cross(r_i-r, f_i)
        return t_center

    def get_force_on_center(self, fragment_indices, f):
        '''
        Get the net force acting on the fragment.

        Inputs:
            fragment_indices: list of length n_atoms_in_fragment
                List containing indices of the fragment's atoms in "full"
            f: np.ndarray of shape (n_atoms_in_full_system, 3)
                Forces acting on the atoms in "full" (in eV/Angstroem)

        Returns:
            np.ndarray of shape (3,)
                Net force acting on the fragment (in eV/Angstroem)
        '''
        fragment_f = f[fragment_indices]
        f_center = np.sum(fragment_f, axis=0)
        return f_center

    def rotate_fragment(self, t_center, stepsize_rot, z_only_rot=False):
        '''
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
        '''
        tmp = self.fragment_inertia_inv@t_center
        if z_only_rot:
            angle = tmp[2] * (180/np.pi) * stepsize_rot 
        else:
            angle = np.linalg.norm(tmp) * (180/np.pi) * stepsize_rot  # in degrees
        if angle != 0:
            if z_only_rot:
                axis = np.array([0.0,0.0,1.0])
            else:
                axis = tmp/np.linalg.norm(tmp)
            self.atoms.rotate(angle,axis,self.atoms.get_center_of_mass())
            self.update_rotation_properties(angle=angle, axis=axis)

        return copy(self.atoms.positions)

    def rotate_fragment2(self, angle, axis):
        '''
        Rotate fragment around com and rotate its axis system and update euler angle
        '''   
        self.atoms.rotate(angle,axis, self.atoms.get_center_of_mass())
        self.update_rotation_properties(angle=angle, axis=axis)
        return copy(self.atoms.positions)


    


    def translate_fragment(self, f_center, stepsize_trans_x, stepsize_trans_y, stepsize_trans_z):
        '''
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
        '''
        fragment_mass = np.sum(self.atoms.get_masses())
        for atom in self.atoms:
            atom.position[0] += stepsize_trans_x * f_center[0]/fragment_mass
            atom.position[1] += stepsize_trans_y * f_center[1]/fragment_mass
            atom.position[2] += stepsize_trans_z * f_center[2]/fragment_mass

        self.update_rotation_properties(angle=0.0, axis=[0.0,0.0,1.0]) #inertia matrix changes due to translation
        return copy(self.atoms.positions)
    
    def copy_fragment(self):
        return deepcopy(self)

    #########################################################################################
    
    def random_step_fragment(self, displacement, angle, seed):
        '''
        Needs to be fixed
        '''
        stepsize_trans_x = 1
        stepsize_trans_y = 1
        stepsize_trans_z = 1
        stepsize_rot     = 1
        fragment_inertia_inv  = np.eye(3)
        fragment_mass = np.sum(self.atoms.get_masses())
        
        backup_seed = np.random.randint(2**32 - 1)
        np.random.seed(seed)
        f_center = np.random.rand(3)
        t_center = np.random.rand(3) 
        np.random.seed(backup_seed)
        
        f_center /= np.linalg.norm(f_center)
        t_center /= np.linalg.norm(t_center)
        f_center *= displacement * fragment_mass
        t_center *= angle * (np.pi/180)
        
        # Translate fragment
        self.translate_fragment(fragment=self.atoms, f_center=f_center, stepsize_trans_x=stepsize_trans_x, stepsize_trans_y=stepsize_trans_y, stepsize_trans_z=stepsize_trans_z)
        # Rotate fragment 
        self.rotate_fragment(fragment=self.atoms, fragment_inertia_inv=fragment_inertia_inv, t_center=t_center, stepsize_rot=stepsize_rot, z_only_rot=False)
        return self.atoms.positions.copy()
    
    def apply_boundaries_fragment(self, xmin, xmax, ymin, ymax):
        '''
        Needs to be fixed
        '''
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
    

    


