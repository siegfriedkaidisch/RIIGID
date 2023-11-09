#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:49:28 2023

@author: kaidisch_siegfried
"""
import numpy as np

from ase import Atom, Atoms

######################################################################################################################

def apply_boundaries_mol(mol, xmin, xmax, ymin, ymax):
    com = mol.get_center_of_mass()
    x = com[0]
    y = com[1]
    print(mol.get_center_of_mass())
    if x < xmin:
        mol.positions[:,0] = 2* xmin - mol.positions[:,0]
    elif x > xmax:
        mol.positions[:,0] = 2* xmax - mol.positions[:,0]

    if y < ymin:
        mol.positions[:,1] = 2* ymin - mol.positions[:,1]
    elif y > ymax:
        mol.positions[:,1] = 2* ymax - mol.positions[:,1]
    print(mol.get_center_of_mass())

    return mol.positions.copy()