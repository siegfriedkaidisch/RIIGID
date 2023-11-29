#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:49:28 2023

Local geometry optimization of rigid fragment on top of rigid surface

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
from copy import deepcopy, copy

from lib.fragment import Fragment

t1 = time.time()


######################################################################################################################
##########################                      Settings                         #####################################
######################################################################################################################

# Settings for rigid geometry optimization
max_rigid_steps = 5 # maximal number of rigid optimization steps
pos_conv = 0.00000000001 # in Angstroem, stop rigid optimization if all atoms of the fragment move less than pos_conv (Warning: may coverge due to small stepsizes rather than actual convergence)
max_step_0 = 0.1 #max allowed change of position of an atom (translation+rotation) in iteration 0
max_step = 0.1 #max allowed change of position of an atom (translation+rotation) in iterations 1+
stepsize_factor_up = 1.2 # increase stepsize by this factor, if last iteration successfully lowered energy
stepsize_factor_dn = 0.2 # decrease stepsize by this factor, if last iteration led to higher energy

# Calculator settings for rigid optimization
rigid_opt_vasp_settings = {
    'directory':'./rigid_opt/',
    'txt':"out",

    'istart':1,
    'npar':4,

    'lorbit':11,
    'lvhar':True,
    'lwave':True,

    'prec':'Accu',
    'encut':300,
    'ediff':1e-7,
    'nelmin':4,
    'nelm':200,
    'algo':'Fast',
#    'enaug':644.9,

    'xc':'PBE',
    'ivdw':12,
    'ismear':0,
    'sigma':0.05,

    'idipol':3,
    'ldipol':True,

#    'ispin':2,
#    'magmom':[0.0]*224 + [1.5],
#    'icharg':1,

    'nsw':0, # leave this at zero!

#    'ldau':True,
#    'ldautype':2,
#    'ldaul':[-1]*5 + [2],
#    'ldauu':[0.0]*5 + [3.0],
#    'ldauj':[0.0]*6,
#    'ldauprint':2,
#    'lmaxmix':4,

    'kpts':[2, 2, 1],
    'gamma':True,  # Gamma-centered k-mesh

    'setups':"recommended"
}

######################################################################################################################
##########################           Prepare Geometry Optimization               #####################################
######################################################################################################################

# Import Geometry
surf = read_vasp(file='./POSCAR_surf')
fragment = read_vasp(file='./POSCAR_fragment')
fragment = Fragment(atoms=fragment)
full = surf + fragment.atoms
#view(full)

# Sort atoms
full = sort(full)

# Export POSCAR for documentation purposes
write_vasp('./POSCAR_start', full, direct=True, vasp5=True, sort=True)

######################################################################################################################
##########################          RIGID Geometry Optimization                  #####################################
######################################################################################################################

# Set up VASP calculator (determines INCAR, KPOINTS and POTCAR)
calculator = Vasp(**rigid_opt_vasp_settings)

# Set calculator
full.set_calculator(calculator)

# Optimize location and rotation of rigid fragment on rigid surface
stepsize = 100 # the value set here doesn't matter, it's just for the first test step, so it will be changed anyway
count_bad_steps = 0 # needed to cope with bad iterations (where energy got larger)
fragment_list = []
energy_list = []
for i in range(max_rigid_steps):
    print('-----')
    print('Iteration ',i)
    #####################################################################
    # Run one SCF cycle
    f = full.get_forces(apply_constraint=False)
    e = full.get_potential_energy(force_consistent=True)
    # Save energy of current geometry
    energy_list.append(e)

    # Needed later (indices of fragment atoms in full system)
    # also, sets the property inside the Fragment object
    fragment_indices = fragment.get_indices(full=full)
    fragment.get_force_on_center(full=full, f=f)
    fragment.get_torque_on_center(full=full, f=f)
    fragment_list.append(deepcopy(fragment))

    #####################################################################

    if i == 0:
        None
    else: # Check if energy got smaller
        if energy_list[-1] < energy_list[-2]:
            # If the prev step successfully lowered the energy, continue as normal and try a bigger stepsize this time
            count_bad_steps = 0
            stepsize *= stepsize_factor_up
        else:
            # Last step led to higher energy -> forget last step and continue from 2nd last step with smaller stepsize
            count_bad_steps +=1  # needed to handle situation of two consecutive bad iterations
            stepsize *= stepsize_factor_dn 
            fragment = deepcopy(fragment_list[-1 - count_bad_steps])
    print('Test Stepsize: ' + str(stepsize))

    # Prepare movement of fragment
    fragment_pos_old = copy(fragment.atoms.positions)

    # Test step to find max displacement of atoms
    fragment_test = deepcopy(fragment)
    fragment_pos_new = fragment_test.transrot_fragment(full=full, f=f, stepsize_trans_x=stepsize, stepsize_trans_y=stepsize, stepsize_trans_z=stepsize,
                    stepsize_rot=stepsize, z_only_rot=False)
    fragment_pos_diff = np.abs(fragment_pos_new - fragment_pos_old)
    fragment_pos_diff_max = np.max(np.linalg.norm(fragment_pos_diff, axis=1))
    del fragment_test

    # Initialize stepsize or adapt it to too large atomic displacements
    if i==0: # In the first iteration we determine the initial stepsize, such that a step of max_step is performed
        factor = max_step_0/fragment_pos_diff_max
    else: # In later iterations, calc factor to prevent too large atomic deplacements
        if fragment_pos_diff_max>max_step:
            factor = max_step/fragment_pos_diff_max
        else:
            factor = 1.0
    stepsize *= factor
    print('Actual Stepsize: ' + str(stepsize))

    # Move fragment and get new positions
    fragment_pos_new = fragment.transrot_fragment(full=full, f=f, stepsize_trans_x=stepsize, stepsize_trans_y=stepsize, stepsize_trans_z=stepsize,
                    stepsize_rot=stepsize, z_only_rot=False)

    del full[fragment_indices]
    full += deepcopy(fragment.atoms)

    #####################################################################

    # Save geometry
    write_vasp('./rigid_opt/CONTCAR_'+str(i), full, direct=True, vasp5=True, sort=True)
    
    # Check for convergence
    fragment_pos_diff = np.abs(fragment_pos_new - fragment_pos_old)
    fragment_pos_diff = np.linalg.norm(fragment_pos_diff, axis=1) 
    if np.max(fragment_pos_diff) < pos_conv:
        print('-----')
        break

######################################################################################################################
##########################                 Finish the job                        #####################################
######################################################################################################################

# Export optimization data from rigid and vasp optim
optim_data = {'energy': energy_list, 'fragment':fragment_list}
f = open('optim_data.pk','wb')
pickle.dump(optim_data, f)
f.close()

print("\n")
print('///////////////////////////////////////////////////////')
print('Summary of Optimization:')
print('Energy:')
print(optim_data['energy'])

print("\n")
print('Final Geometry:')
print("Energy: "+str(optim_data['energy'][-1]))

print("\n")
print('Duration / h:')
print((time.time()-t1)/3600)
print('///////////////////////////////////////////////////////')
