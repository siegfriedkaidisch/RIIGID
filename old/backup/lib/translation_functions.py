#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:49:28 2023

@author: kaidisch_siegfried
"""
import numpy as np

from ase import Atom, Atoms

######################################################################################################################

def get_force_mol(mol_indices, f):
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
    mol_f = f[mol_indices]
    f_center = np.sum(mol_f, axis=0)
    return f_center

######################################################################################################################

def translate_mol(mol, f_center, stepsize_trans_x, stepsize_trans_y, stepsize_trans_z):
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
    mol_mass = np.sum(mol.get_masses())
    for atom in mol:
        atom.position[0] += stepsize_trans_x * f_center[0]/mol_mass
        atom.position[1] += stepsize_trans_y * f_center[1]/mol_mass
        atom.position[2] += stepsize_trans_z * f_center[2]/mol_mass
    return mol.positions.copy()

######################################################################################################################
