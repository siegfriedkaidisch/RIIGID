#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 14:28:48 2023

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

###############################################################################
max_dist_between_atoms = 0.05 #Å
path_1  = '/home/kaidisch_siegfried/Documents/VASP/rigid_optim/compare_e_surf/surf_rigid/rot_20/1-0/test_speedup1/optim_data.pk'
path_2 = '/home/kaidisch_siegfried/Documents/VASP/rigid_optim/compare_e_surf/surf_rigid/rot_40/1-0/test_speedup1/optim_data.pk'

f = open(path_1, 'rb')
data_to_compare_to = pickle.load(f)
f.close()

f = open(path_2, 'rb')
data_to_be_compared = pickle.load(f)
f.close()

# Note: rot 20 and 40 have quasi same result, while 0 is different
###############################################################################
data_to_compare_to  = data_to_compare_to['mol']
data_to_be_compared = data_to_be_compared['mol']

for i, mol_to_be_compared in enumerate(data_to_be_compared):
    print('Step: ',i)
    found = False
    for mol_to_compare_to in data_to_compare_to:
        diff_positions = mol_to_be_compared.positions - mol_to_compare_to.positions
        diff_positions = np.linalg.norm(diff_positions, axis=1) 
        max_diff       = max(diff_positions)
        if max_diff < max_dist_between_atoms:
            print('FOUND!')
            found = True
            break
    if found:
        break
        
        """
        min_diffs_mol = []
        for atom_to_be_compared in mol_to_be_compared:
            min_diff_atom = 1e10
            for atom_to_compare_to in mol_to_compare_to:
                # additionally should check type of atoms to be the same...
                diff_atom = atom_to_be_compared.position - atom_to_compare_to.position
                diff_atom = np.linalg.norm(diff_atom) 
                if diff_atom < min_diff_atom:
                    min_diff_atom = diff_atom
            min_diffs_mol.append(min_diff_atom)
        max_min_diff_mol = max(min_diffs_mol)
        if max_min_diff_mol < max_dist_between_atoms:
            print('FOUND!')
            break
        """













