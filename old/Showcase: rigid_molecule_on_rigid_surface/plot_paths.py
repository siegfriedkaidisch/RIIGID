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
            pos_z = np.array([(tmp[2]) for tmp in positions])

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


norm_z = plt.Normalize(min_z, max_z)
norm_alpha = plt.Normalize(min_alpha, max_alpha)
norm_beta  = plt.Normalize(min_beta, max_beta)
norm_gamma = plt.Normalize(min_gamma, max_gamma)
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

a = 3.568
img = plt.imread("background.png")
for i in [0,1]:
    for j in [0,1]:
        axs[i,j].imshow(img, aspect='equal', extent=[-a,a,-a,a])


for rot in [0,20,40]:
    for x in [0,1]:
        for y in [0,1]:
            path = '/home/kaidisch_siegfried/Documents/VASP/rigid_optim/compare_e_surf/surf_rigid/rot_'+str(rot)+'/'+str(x)+'-'+str(y)+'/test_speedup1/optim_data.pk'

            f = open(path, 'rb')
            data = pickle.load(f)
            f.close()
            
            positions = data['position']
            pos_x = np.array([(tmp[0]) for tmp in positions])
            pos_y = np.array([(tmp[1]) for tmp in positions])
            pos_z = np.array([(tmp[2]) for tmp in positions])

            angles = data['rotation']
            alpha = np.array([tmp[0] for tmp in angles])
            beta  = np.array([tmp[1] for tmp in angles])
            gamma = np.array([tmp[2] for tmp in angles])


            # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/multicolored_line.html
            # Create a set of line segments so that we can color them individually
            # This creates the points as a N x 1 x 2 array so that we can stack points
            # together easily to get the segments. The segments array for line collection
            # needs to be (numlines) x (points per line) x 2 (for x and y)
            points = np.array([pos_x, pos_y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)


            id = -1
            cmap = 'hot'
            # Set the values used for colormapping
            lc_z = LineCollection(segments, cmap=cmap, norm=norm_z)
            lc_z.set_array(pos_z)
            lc_z.set_linewidth(2)
            line_z = axs[0,0].add_collection(lc_z)
            cm = line_z.cmap
            axs[0,0].scatter(pos_x[id],pos_y[id],s=100.0, c = cm(norm_z(pos_z[id])), marker='X')

            lc_alpha = LineCollection(segments, cmap=cmap, norm=norm_alpha)
            lc_alpha.set_array(alpha)
            lc_alpha.set_linewidth(2)
            line_alpha = axs[0,1].add_collection(lc_alpha)
            cm = line_alpha.cmap
            axs[0,1].scatter(pos_x[id],pos_y[id],s=100.0, c = cm(norm_alpha(alpha[id])), marker='X')

            lc_beta = LineCollection(segments, cmap=cmap, norm=norm_beta)
            lc_beta.set_array(beta)
            lc_beta.set_linewidth(2)
            line_beta = axs[1,0].add_collection(lc_beta)
            cm = line_beta.cmap
            axs[1,0].scatter(pos_x[id],pos_y[id],s=100.0, c = cm(norm_beta(beta[id])), marker='X')

            lc_gamma = LineCollection(segments, cmap=cmap, norm=norm_gamma)
            lc_gamma.set_array(gamma)
            lc_gamma.set_linewidth(2)
            line_gamma = axs[1,1].add_collection(lc_gamma)
            cm = line_gamma.cmap
            axs[1,1].scatter(pos_x[id],pos_y[id],s=100.0, c = cm(norm_gamma(gamma[id])), marker='X')


fig.colorbar(line_z, ax=axs[0,0])
fig.colorbar(line_alpha, ax=axs[0,1])
fig.colorbar(line_beta, ax=axs[1,0])
fig.colorbar(line_gamma, ax=axs[1,1])

for i in [0,1]:
    for j in [0,1]:
        axs[i,j].set_xlim([-a,a])
        axs[i,j].set_ylim([-a,a])
        axs[i,j].set_xlabel("x / Å")
        axs[i,j].set_ylabel("y / Å")


plt.show()
            
            
















