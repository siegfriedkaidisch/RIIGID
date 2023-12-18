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

class Rigid_Atoms():
    def __init__(self, atoms:Atoms):
        # Initialize using an already existing (normal) Atoms object
        self.atoms = deepcopy(atoms)
        del atoms

        # Body-fixed axis system, euler angles and inertia matrix
        self.xb = np.array([1.0,0.0,0.0])
        self.yb = np.array([0.0,1.0,0.0])
        self.zb = np.array([0.0,0.0,1.0])
        self.euler_angles = [0.0,0.0,0.0]
        self.update_inertia_mat_and_inv()

        return None


    def update_euler_angles(self, xs=[1,0,0],ys=[0,1,0],zs=[0,0,1]):
        '''
        body-fixed vs space-fixed axis system (body-axes given in space-coords)
        convention: if z axes are parallel, set xs=N
        '''
        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)
        
        beta  = angle_between_vectors(v1=zs, v2=self.zb)
        
        if beta == 0 or beta==180:
            alpha = 0.0
            gamma = angle_between_vectors2(v1=xs,v2=self.xb, axis=self.zb)
        else:
            N = np.cross(zs, self.zb)
            alpha = angle_between_vectors2(v1=xs,v2=N, axis=zs)
            gamma = angle_between_vectors2(v1=N,v2=self.xb, axis=self.zb)
        
        self.euler_angles = [alpha, beta, gamma]

        return copy(self.euler_angles)
    
    def update_rot_props_mol(self, angle, axis):
        mat = rotmat(axis=axis,angle=angle)
        self.xb = mat@self.xb
        self.yb = mat@self.yb
        self.zb = mat@self.zb
        self.update_euler_angles()
        self.update_inertia_mat_and_inv()
        return None

    def update_inertia_mat_and_inv(self):
        '''
        Get inertia tensor (and its inverse) of a molecule.

        Inputs:
            mol: ase.atoms.Atoms
                The molecule whose inertia matrix (and inverse) will be calculated

        Returns:
            numpy.ndarray of shape (3,3)
                The inertia matrix of the molecule (in Da*Å**2)
            numpy.ndarray of shape (3,3)
                The inverse inertia matrix of the molecule (in 1/(Da*Å**2))
        '''
        mol_com = self.atoms.get_center_of_mass()
        mol_inertia = np.zeros([3,3])
        for j in range(3):
            for k in range(3):
                for atom in self.atoms:
                    r_l = atom.position - mol_com
                    if j==k:
                        mol_inertia[j,k] += atom.mass * ( np.linalg.norm(r_l)**2 - r_l[j]*r_l[k] ) 
                    else:
                        mol_inertia[j,k] += atom.mass * (                - r_l[j]*r_l[k] ) 
        mol_inertia_inv = np.linalg.inv(mol_inertia)

        self.mol_inertia = mol_inertia
        self.mol_inertia_inv = mol_inertia_inv
        return copy(mol_inertia), copy(mol_inertia_inv)
    
    def get_mol_indices2(self, full, cutoff=1e-4):
        mol_indices = []
        for mol_atom in self.atoms:
            for full_atom in full:
                if np.linalg.norm(mol_atom.position - full_atom.position) < cutoff:
                    mol_indices.append(full_atom.index)
        self.mol_indices = mol_indices
        return copy(mol_indices)

    def get_torque_mol(self, full, f):
        '''
        Get the net torque acting on the molecule (relative to its center of mass).

        Inputs:
            full: ase.atoms.Atoms
                The full system (surface+molecule) under study
            mol_indices: list of length n_atoms_in_molecule
                List containing indices of the molecule's atoms in "full"
            f: numpy.ndarray of shape (n_atoms_in_full_system, 3)
                Forces acting on the atoms in "full" (in eV/Å)

        Returns:
            numpy.ndarray of shape (3,)
                Net torque acting on the molecule (relative to center of mass of molecule) (in eV)
        '''
        mol_com = self.atoms.get_center_of_mass()
        t_center = np.array([0.0,0.0,0.0])
        for index in self.mol_indices:
            r_i = full[index].position
            r   = mol_com
            f_i = f[index]
            t_center += np.cross(r_i-r, f_i)
        self.t_center = t_center
        return copy(t_center)

    def rotate_mol(self, full, f, stepsize_rot, z_only_rot=False):
        '''
        Rotate molecule around its center of mass following the applied torque

        Inputs:
            mol: ase.atoms.Atoms
                The molecule to be rotated
            mol_inertia_inv: numpy.ndarray of shape (3,3)
                The inverse inertia matrix of the molecule (in 1/(Da*Å**2))
            t_center: numpy.ndarray of shape (3,) or list of length 3
                Net torque acting on the molecule (relative to center of mass of molecule) (in eV)
            stepsize_rot: number
                Timestep (in Da*Å**2/eV)
            z_only_rot: Bool
                Rotate only around the z-axis?

        Returns:
            numpy.ndarray of shape (n_atoms_in_molecule,3)
                The positions (in Å) of the molecule's atoms after the transformation
        '''
        self.update_inertia_mat_and_inv()

        tmp = self.mol_inertia_inv@self.t_center
        if z_only_rot:
            angle = tmp[2] * (180/np.pi) * stepsize_rot 
        else:
            angle = np.linalg.norm(tmp) * (180/np.pi) * stepsize_rot  # in °
        if angle != 0:
            if z_only_rot:
                axis = np.array([0.0,0.0,1.0])
            else:
                axis = tmp/np.linalg.norm(tmp)
            self.atoms.rotate(angle,axis,self.atoms.get_center_of_mass())
            self.update_rot_props_mol(angle=angle, axis=axis)

        return copy(self.atoms.positions)

    def rotate_mol2(self, angle, axis):
        '''
        Rotate mol around com and rotate its axis system and update euler angle
        '''   
        self.atoms.rotate(angle,axis, self.atoms.get_center_of_mass())
        self.update_rot_props_mol(angle=angle, axis=axis)
        return copy(self.atoms.positions)

    def get_force_mol(self, full, f):
        '''
        Get the net force acting on the molecule.

        Inputs:
            mol_indices: list of length n_atoms_in_molecule
                List containing indices of the molecule's atoms in "full"
            f: numpy.ndarray of shape (n_atoms_in_full_system, 3)
                Forces acting on the atoms in "full" (in eV/Å)

        Returns:
            numpy.ndarray of shape (3,)
                Net force acting on the molecule (in eV/Å)
        '''
        mol_f = f[self.mol_indices]
        f_center = np.sum(mol_f, axis=0)
        self.f_center = f_center
        return copy(f_center)
    
    def translate_mol(self, full, f, stepsize_trans_x, stepsize_trans_y, stepsize_trans_z):
        '''
        Translate molecule in direction of the applied force

        Inputs:
            mol: ase.atoms.Atoms
                The molecule to be translated
            f_center: numpy.ndarray of shape (3,) or list of length 3
                Net force acting on the molecule (in eV/Å)
            stepsize_trans_x/y/z: number
                Timesteps; usually all three should have the same value; (in Da*Å**2/eV)

        Returns:
            numpy.ndarray of shape (n_atoms_in_molecule,3)
                The positions (in Å) of the molecule's atoms after the transformation
        '''
        mol_mass = np.sum(self.atoms.get_masses())
        for atom in self.atoms:
            atom.position[0] += stepsize_trans_x * self.f_center[0]/mol_mass
            atom.position[1] += stepsize_trans_y * self.f_center[1]/mol_mass
            atom.position[2] += stepsize_trans_z * self.f_center[2]/mol_mass

        self.update_rot_props_mol(self, angle=0.0, axis=[0.0,0.0,1.0]) #inertia matrix changes due to translation
        return copy(self.atoms.positions)
    
    def transrot_mol(self, full, f, stepsize_trans_x, stepsize_trans_y, stepsize_trans_z, 
                    stepsize_rot, z_only_rot=False):
        '''
        Translate molecule in direction of the applied force and rotate it around its center of mass following the applied torque

        Inputs:
            mol: ase.atoms.Atoms
                The molecule to be translated
            f_center: numpy.ndarray of shape (3,) or list of length 3
                Net force acting on the molecule (in eV/Å)
            stepsize_trans_x/y/z/rot: number
                Timesteps; usually all four should have the same value; (in Da*Å**2/eV)
            mol_inertia_inv: numpy.ndarray of shape (3,3)
                The inverse inertia matrix of the molecule (in 1/(Da*Å**2))
            t_center: numpy.ndarray of shape (3,) or list of length 3
                Net torque acting on the molecule (relative to center of mass of molecule) (in eV)
            z_only_rot: Bool
                Rotate only around the z-axis?

        Returns:
            numpy.ndarray of shape (n_atoms_in_molecule,3)
                The positions (in Å) of the molecule's atoms after the transformation
        '''
        # Translate molecule
        self.translate_mol(full=full, f=f, stepsize_trans_x=stepsize_trans_x, stepsize_trans_y=stepsize_trans_y, stepsize_trans_z=stepsize_trans_z)
        # Rotate molecule 
        self.rotate_mol(full=full, f=f, stepsize_rot=stepsize_rot, z_only_rot=z_only_rot)
        return copy(self.atoms.positions)
    
    def copy_mol(self):
        return deepcopy(self)

    #########################################################################################
    
    def random_step_mol(self, displacement, angle, seed):
        '''
        Needs to be fixed
        '''
        stepsize_trans_x = 1
        stepsize_trans_y = 1
        stepsize_trans_z = 1
        stepsize_rot     = 1
        mol_inertia_inv  = np.eye(3)
        mol_mass = np.sum(self.atoms.get_masses())
        
        backup_seed = np.random.randint(2**32 - 1)
        np.random.seed(seed)
        f_center = np.random.rand(3)
        t_center = np.random.rand(3) 
        np.random.seed(backup_seed)
        
        f_center /= np.linalg.norm(f_center)
        t_center /= np.linalg.norm(t_center)
        f_center *= displacement * mol_mass
        t_center *= angle * (np.pi/180)
        
        # Translate molecule
        self.translate_mol(mol=self.atoms, f_center=f_center, stepsize_trans_x=stepsize_trans_x, stepsize_trans_y=stepsize_trans_y, stepsize_trans_z=stepsize_trans_z)
        # Rotate molecule 
        self.rotate_mol(mol=self.atoms, mol_inertia_inv=mol_inertia_inv, t_center=t_center, stepsize_rot=stepsize_rot, z_only_rot=False)
        return self.atoms.positions.copy()
    
    def apply_boundaries_mol(self, xmin, xmax, ymin, ymax):
        '''
        Needs to be fixed
        '''
        com = self.atoms.get_center_of_mass()
        x = com[0]
        y = com[1]
        #print(mol.get_center_of_mass())
        if x < xmin:
            self.atoms.positions[:,0] = 2* xmin - self.atoms.positions[:,0]
        elif x > xmax:
            self.atoms.positions[:,0] = 2* xmax - self.atoms.positions[:,0]

        if y < ymin:
            self.atoms.positions[:,1] = 2* ymin - self.atoms.positions[:,1]
        elif y > ymax:
            self.atoms.positions[:,1] = 2* ymax - self.atoms.positions[:,1]
        #print(mol.get_center_of_mass())

        return self.atoms.positions.copy()
    

    


