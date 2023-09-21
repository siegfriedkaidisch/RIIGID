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

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

###############################################################################

min_z = 1000.0
max_z = 0.0
min_alpha = 1000.0
max_alpha = 0.0
min_beta = 1000.0
max_beta = 0.0
min_gamma = 1000.0
max_gamma = 0.0
for rot in [0,20,40]:
    for x in [0,1]:
        for y in [0,1]:
            path = '/home/kaidisch_siegfried/Documents/VASP/rigid_optim/compare_e_surf/surf_rigid/rot_'+str(rot)+'/'+str(x)+'-'+str(y)+'/test_speedup1/optim_data.pk'

            f = open(path, 'rb')
            data = pickle.load(f)
            f.close()
            
            positions = data['position']
            pos_z = np.array([np.abs(tmp[2]) for tmp in positions])

            if pos_z.max() > max_z:
                max_z = pos_z.max()

            if pos_z.min() < min_z:
                min_z = pos_z.min()

            angles = data['rotation']
            alpha = np.array([tmp[0] for tmp in angles])
            beta  = np.array([tmp[1] for tmp in angles])
            gamma = np.array([tmp[2] for tmp in angles])

            if alpha.max() > max_alpha:
                max_alpha = alpha.max()

            if alpha.min() < min_alpha:
                min_alpha = alpha.min()

            if beta.max() > max_beta:
                max_beta = beta.max()

            if beta.min() < min_beta:
                min_beta = beta.min()

            if gamma.max() > max_gamma:
                max_gamma = gamma.max()

            if gamma.min() < min_gamma:
                min_gamma = gamma.min()


norm = plt.Normalize(min_z, max_z)
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
for rot in [0,20,40]:
    for x in [0,1]:
        for y in [0,1]:
            path = '/home/kaidisch_siegfried/Documents/VASP/rigid_optim/compare_e_surf/surf_rigid/rot_'+str(rot)+'/'+str(x)+'-'+str(y)+'/test_speedup1/optim_data.pk'

            f = open(path, 'rb')
            data = pickle.load(f)
            f.close()
            
            positions = data['position']
            pos_x = np.array([np.abs(tmp[0]) for tmp in positions])
            pos_y = np.array([np.abs(tmp[1]) for tmp in positions])
            pos_z = np.array([np.abs(tmp[2]) for tmp in positions])


            # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/multicolored_line.html
            # Create a set of line segments so that we can color them individually
            # This creates the points as a N x 1 x 2 array so that we can stack points
            # together easily to get the segments. The segments array for line collection
            # needs to be (numlines) x (points per line) x 2 (for x and y)
            points = np.array([pos_x, pos_y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Create a continuous norm to map from data points to colors
            lc = LineCollection(segments, cmap='viridis', norm=norm)
            # Set the values used for colormapping
            lc.set_array(pos_z)
            lc.set_linewidth(2)
            line = ax.add_collection(lc)
            cm = line.cmap
            n  = line.norm

            id = -1
            plt.scatter(pos_x[id],pos_y[id],s=100.0, c = cm(norm(pos_z[id])), marker='X')


fig.colorbar(line, ax=ax)
plt.xlim([-0.1,2.6])
plt.ylim([-0.1,1.1])
plt.xlabel("x / AA")
plt.ylabel("y / AA")
plt.show()
            
            
















