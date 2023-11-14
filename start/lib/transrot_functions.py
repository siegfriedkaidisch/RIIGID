#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:49:28 2023

@author: kaidisch_siegfried
"""
import numpy as np

from ase import Atom, Atoms

from lib.translation_functions import translate_mol
from lib.rotation_functions import rotate_mol

######################################################################################################################

def transrot_mol(mol, f_center, stepsize_trans_x, stepsize_trans_y, stepsize_trans_z, 
                    mol_inertia_inv, t_center, stepsize_rot, z_only_rot=False):
    '''
    Translate molecule in direction of the applied force and rotate it around its center of mass following the applied torque

    Inputs:
        mol: ase.atoms.Atoms
            The molecule to be translated
        f_center: np.ndarray of shape (3,) or list of length 3
            Net force acting on the molecule (in eV/Angstroem)
        stepsize_trans_x/y/z/rot: number
            Timesteps; usually all four should have the same value; (in Dalton*Angstroem**2/eV)
        mol_inertia_inv: np.ndarray of shape (3,3)
            The inverse inertia matrix of the molecule (in 1/(Dalton*Angstroem**2))
        t_center: np.ndarray of shape (3,) or list of length 3
            Net torque acting on the molecule (relative to center of mass of molecule) (in eV)
        z_only_rot: Bool
            Rotate only around the z-axis?

    Returns:
        np.ndarray of shape (n_atoms_in_molecule,3)
            The positions (in Angstroem) of the molecule's atoms after the transformation
    '''
    # Translate molecule
    translate_mol(mol=mol, f_center=f_center, stepsize_trans_x=stepsize_trans_x, stepsize_trans_y=stepsize_trans_y, stepsize_trans_z=stepsize_trans_z)
    # Rotate molecule 
    rotate_mol(mol=mol, mol_inertia_inv=mol_inertia_inv, t_center=t_center, stepsize_rot=stepsize_rot, z_only_rot=z_only_rot)
    return mol.positions.copy()

######################################################################################################################

def random_step_mol(mol, displacement, angle, seed):
    '''

    '''
    stepsize_trans_x = 1
    stepsize_trans_y = 1
    stepsize_trans_z = 1
    stepsize_rot     = 1
    mol_inertia_inv  = np.eye(3)
    mol_mass = np.sum(mol.get_masses())
    
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
    translate_mol(mol=mol, f_center=f_center, stepsize_trans_x=stepsize_trans_x, stepsize_trans_y=stepsize_trans_y, stepsize_trans_z=stepsize_trans_z)
    # Rotate molecule 
    rotate_mol(mol=mol, mol_inertia_inv=mol_inertia_inv, t_center=t_center, stepsize_rot=stepsize_rot, z_only_rot=False)
    return mol.positions.copy()

######################################################################################################################
