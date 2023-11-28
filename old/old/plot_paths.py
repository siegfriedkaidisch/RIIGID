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
import matplotlib.pyplot as plt

###############################################################################

cm = plt.get_cmap('gist_rainbow')
maxrot = 60
for rot in [40]:
    for x in [0]:
        for y in [1]:
            color = rot/maxrot
            path = '/home/kaidisch_siegfried/Documents/VASP/rigid_optim/compare_e_surf/surf_rigid/rot_'+str(rot)+'/'+str(x)+'-'+str(y)+'/test_speedup1/optim_data.pk'

            f = open(path, 'rb')
            data = pickle.load(f)
            f.close()
            
            positions = data['position']
            pos_x = [(tmp[0]) for tmp in positions]
            pos_y = [(tmp[1]) for tmp in positions]
            plt.plot(pos_x,pos_y, c=cm(color))
            plt.scatter(pos_x[-1],pos_y[-1],s=100.0, c=cm(color), marker='X')
            

plt.xlabel("x / AA")
plt.ylabel("y / AA")
plt.show()
            















