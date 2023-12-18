#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fr Nov 24 2023

Python Class for Rigid Atoms

@author: kaidisch_siegfried
"""
import numpy as np
import pickle
import time

from ase import Atom, Atoms
from ase.build import add_adsorbate, make_supercell, sort, add_vacuum
from ase.visualize import view
from ase.io import read
from ase.io.vasp import write_vasp
from ase.constraints import FixAtoms, FixBondLengths
from ase.io.vasp import read_vasp
import os
from ase.calculators.vasp.vasp import Vasp
from ase.optimize import BFGS, MDMin
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG

from lib.rotation_functions import angle_between_vectors, angle_between_vectors2, rotmat

class Rigid_Atoms():
    def __init__(self, atoms:Atoms):
        # Initialize using an already existing (normal) Atoms object
        self.atoms = atoms.copy()

        # Body-fixed axis system and euler angles
        self.xb = np.array([1.0,0.0,0.0])
        self.yb = np.array([0.0,1.0,0.0])
        self.zb = np.array([0.0,0.0,1.0])
        self.euler_angles = [0.0,0.0,0.0]

        return None


    def get_euler_angles(self, xs=[1,0,0],ys=[0,1,0],zs=[0,0,1]):
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

        return self.euler_angles
    
    def update_rot_props_mol(self, angle, axis):
        # better to just include in rotation method?
        mat = rotmat(axis=axis,angle=angle)
        self.xb = mat@self.xb
        self.yb = mat@self.yb
        self.zb = mat@self.zb
        self.get_euler_angles()
        return None

    def get_inertia_mat_and_inv(self):
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
        return mol_inertia, mol_inertia_inv
    
    def get_mol_indices2(self, full, cutoff=1e-4):
        mol_indices = []
        for mol_atom in self.atoms:
            for full_atom in full:
                if np.linalg.norm(mol_atom.position - full_atom.position) < cutoff:
                    mol_indices.append(full_atom.index)
        return mol_indices

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
        mol_indices = self.get_mol_indices2(full=full)
        mol_com = full[mol_indices].get_center_of_mass()
        t_center = np.array([0.0,0.0,0.0])
        for index in mol_indices:
            r_i = full[index].position
            r   = mol_com
            f_i = f[index]
            t_center += np.cross(r_i-r, f_i)
        self.t_center = t_center
        return t_center

    def rotate_mol(self, stepsize_rot, z_only_rot=False):
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
        t_center = np.array(self.t_center)
        mol_com = self.atoms.get_center_of_mass()
        tmp = self.mol_inertia_inv@t_center
        if z_only_rot:
            angle = tmp[2] * (180/np.pi) * stepsize_rot 
        else:
            angle = np.linalg.norm(tmp) * (180/np.pi) * stepsize_rot  # in °
        if angle != 0:
            if z_only_rot:
                axis = np.array([0.0,0.0,1.0])
            else:
                axis = tmp/np.linalg.norm(tmp)
            center = mol_com
            self.atoms.rotate(angle,axis,center)
            self.update_rot_props_mol(angle=angle, axis=axis)

        return self.atoms.positions.copy()

    def rotate_mol2(self, angle, axis):
        '''
        Rotate mol around com and rotate its axis system and update euler angle
        '''   
        self.atoms.rotate(angle,axis, self.atoms.get_center_of_mass())
        self.update_rot_props_mol(angle=angle, axis=axis)
        return None
