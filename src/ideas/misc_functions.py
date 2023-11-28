#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:49:28 2023

@author: kaidisch_siegfried
"""
import numpy as np

from ase import Atom, Atoms


######################################################################################################################

def get_mol_indices(full, middle_height, above=True):
    '''
    Given the full system ("full"=surface+molecule), find the indices of the molecule's atoms. 
    To do so, specifiy a height ("middle_height") separating the surface from the molecule. 

    above=True: Atoms below middle_height are considered to belong to the surface and atoms above middle_height 
    are considered to belong to the molecule.

    above=False: Atoms above middle_height are considered to belong to the surface and atoms below middle_height 
    are considered to belong to the molecule.

    Inputs:
        full: ase.atoms.Atoms
            The full system (surface+molecule) under study
        middle_height: number
            Height (in Angstroem) used to separate molecule from surface (see description above)
        above: Bool
            See explanation given above

    Returns:
        list of length n_atoms_in_molecule
            List containing indices of the molecule's atoms in "full"
    '''
    if above:
        return [ atom.index for atom in full if atom.position[2] >= middle_height ]
    else:
        return [ atom.index for atom in full if atom.position[2] < middle_height ]


######################################################################################################################
