#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:49:28 2023

Local geometry optimization of rigid molecule on top of rigid surface

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

from lib.rotation_functions import add_rot_props_mol, rotate_mol2, get_inertia_mat_and_inv, get_torque_mol
from lib.translation_functions import get_force_mol
from lib.transrot_functions import random_step_mol, transrot_mol
from lib.boundary_functions import apply_boundaries_mol
from lib.misc_functions import get_mol_indices, copy_mol
    
t1 = time.time()
######################################################################################################################
##########################                      Settings                         #####################################
######################################################################################################################

# General settings
middle_height = 9.0 #in Angstroem, used to separate molecule and surface
vacuum = 20 # in Angstroem, total height of vacuum
a = 3.568 # in Angstroem, lattice parameter

# Control initial geometry of molecule relative to surface
pos_x = 0 * a/5 # Angstroem
pos_y = 0 * a/5 # Angstroem
rot_z = 2 * 20  # degrees

# Settings for rigid geometry optimization
max_rigid_steps = 500 # maximal number of rigid optimization steps
pos_conv = 0.00000000001 # in Angstroem, stop rigid optimization if all atoms of the molecule move less than pos_conv (Warning: may coverge due to small stepsizes rather than actual convergence)
max_step_0 = 0.01 #max allowed change of position of an atom (translation+rotation) in iteration 0
max_step = 0.1 #max allowed change of position of an atom (translation+rotation) in iterations 1+
stepsize_factor_up = 1.2 # increase stepsize by this factor, if last iteration successfully lowered energy
stepsize_factor_dn = 0.2 # decrease stepsize by this factor, if last iteration led to higher energy

# Randomness used to escape saddle points (at the moment, only used to escape saddle point initial guesses)
do_random_step      = True
random_seed         = 12345
random_displacement = 0.02 #Angstroem
random_angle        = 0.5 #Degrees

# Boundary conditions
xmin = 0
xmax = a
ymin = 0
ymax = a/2

# Settings for "normal" VASP geometry optimization following the rigid optimization
do_vasp_opt = False
freeze_height = 3 # in Angstroem, freeze atoms below this height

# Calculator settings for rigid optimization
rigid_opt_vasp_settings = {
    'directory':'./rigid_opt/',
    'txt':"out",

    'istart':0,
    'npar':4,

    'lorbit':11,
    'lvhar':True,
    'lwave':True,

    'prec':'Accu',
    'encut':400,
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

# Calculator settings for normal VASP optimization
normal_opt_vasp_settings = rigid_opt_vasp_settings.copy()
normal_opt_vasp_settings['directory'] = './vasp_opt/'
normal_opt_vasp_settings['istart'] = 1
normal_opt_vasp_settings['ibrion'] = 2
normal_opt_vasp_settings['ediffg'] = -1e-2
normal_opt_vasp_settings['nsw'] = 1000

######################################################################################################################
##########################           Prepare Geometry Optimization               #####################################
######################################################################################################################

# Import Geometry
full = read_vasp(file='./POSCAR_input')
# Control initial geometry
mol_indices = get_mol_indices(full=full, middle_height=middle_height)
mol = full[mol_indices].copy()
add_rot_props_mol(mol)
del full[mol_indices]
mol.positions[:,0] += -mol.get_center_of_mass()[0] + pos_x
mol.positions[:,1] += -mol.get_center_of_mass()[1] + pos_y
rotate_mol2(mol=mol, angle=rot_z, axis=[0.0,0.0,1.0])
if do_random_step:
    # Random step to get rid of symmetry in starting point
    random_step_mol(mol=mol, displacement=random_displacement, angle=random_angle, seed=random_seed)
apply_boundaries_mol(mol=mol, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
full += mol.copy()

# Sort atoms
full = sort(full)

# Freeze (some) surface atoms (only relevant for VASP optimization)
c1 = FixAtoms(indices=[ atom.index for atom in full if atom.position[2]<freeze_height ])

# Apply Contraints
del full.constraints
full.set_constraint(c1)

# Fix vacuum height
existing_vac = full.get_cell_lengths_and_angles()[2] - np.max(full.positions[:,2]) + np.min(full.positions[:,2])
add_vacuum(full, vacuum-existing_vac)

# Export POSCAR for documentation purposes
write_vasp('./POSCAR_start', full, direct=True, vasp5=True, sort=True)

######################################################################################################################
##########################          RIGID Geometry Optimization                  #####################################
######################################################################################################################

# Set up VASP calculator (determines INCAR, KPOINTS and POTCAR)
calculator = Vasp(**rigid_opt_vasp_settings)

# Set calculator
full.set_calculator(calculator)

# Optimize location and rotation of rigid molecule on rigid surface
stepsize = 100 # the value set here doesn't matter, it's just for the first test step, so it will be changed anyway
count_bad_steps = 0 # needed to cope with bad iterations (where energy got larger)
mol_list = []
pos_list = []
mol_inertia_list = []
mol_inertia_inv_list = []
rot_list = []
energy_list = []
f_center_list = []
t_center_list = []
for i in range(max_rigid_steps):
    print('-----')
    print('Iteration ',i)
    #####################################################################

    # Save for later
    mol_list.append(copy_mol(mol))

    # Get indices of the atoms forming the molecule
    mol_indices = get_mol_indices(full=full, middle_height=middle_height)

    # Save center of mass of current geometry
    mol_com = mol.get_center_of_mass()
    pos_list.append(mol_com)

    # Get inertia matrix and inverse
    mol_inertia, mol_inertia_inv = get_inertia_mat_and_inv(mol=full[mol_indices])
    mol_inertia_list.append(mol_inertia)
    mol_inertia_inv_list.append(mol_inertia_inv)

    # Save rotation angles of current geometry
    rot_list.append(mol.euler_angles)
    
    # Run one SCF cycle
    f = full.get_forces(apply_constraint=False)
    e = full.get_potential_energy(force_consistent=True)

    # Save energy of current geometry
    energy_list.append(e)

    # Get net force on molecule
    f_center = get_force_mol(mol_indices=mol_indices, f=f)
    f_center_list.append(f_center)

    # get torque on molecule
    t_center = get_torque_mol(full=full, mol_indices=mol_indices, f=f)
    t_center_list.append(t_center)

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
            mol = copy_mol(mol_list[-1 - count_bad_steps])
            f_center = f_center_list[-1 - count_bad_steps]
            mol_inertia_inv = mol_inertia_inv_list[-1 - count_bad_steps]
            t_center = t_center_list[-1 - count_bad_steps]
    print('Test Stepsize: ' + str(stepsize))

    # Prepare movement of molecule
    del full[mol_indices]
    mol_pos_old = mol.positions.copy()

    # Test step to find max displacement of atoms
    mol_test = copy_mol(mol)
    mol_pos_new = transrot_mol(mol=mol_test, f_center=f_center, stepsize_trans_x=stepsize, stepsize_trans_y=stepsize, stepsize_trans_z=stepsize, 
                                        mol_inertia_inv=mol_inertia_inv, t_center=t_center, stepsize_rot=stepsize)
    mol_pos_diff = np.abs(mol_pos_new - mol_pos_old)
    mol_pos_diff_max = np.max(np.linalg.norm(mol_pos_diff, axis=1))
    del mol_test

    # Initialize stepsize or adapt it to too large atomic displacements
    if i==0: # In the first iteration we determine the initial stepsize, such that a step of max_step is performed
        factor = max_step_0/mol_pos_diff_max
    else: # In later iterations, calc factor to prevent too large atomic deplacements
        if mol_pos_diff_max>max_step:
            factor = max_step/mol_pos_diff_max
        else:
            factor = 1.0
    stepsize *= factor
    print('Actual Stepsize: ' + str(stepsize))

    # Move molecule and get new positions
    mol_pos_new = transrot_mol(mol=mol, f_center=f_center, stepsize_trans_x=stepsize, stepsize_trans_y=stepsize, stepsize_trans_z=stepsize, 
                                        mol_inertia_inv=mol_inertia_inv, t_center=t_center, stepsize_rot=stepsize)

    # Apply boundary conditions
    apply_boundaries_mol(mol=mol, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)    

    full += mol.copy()

    #####################################################################

    # Print some stuff for testing purposes
    print('Displacement/stepsize: ' + str(f_center/np.sum(mol.get_masses())))
    print('Angle/stepsize: ' + str(np.linalg.norm(mol_inertia_inv@t_center) * (180/np.pi)))
    print('-----')

    # Save geometry
    write_vasp('./rigid_opt/CONTCAR_'+str(i), full, direct=True, vasp5=True, sort=True)
    
    # Check for convergence
    mol_pos_diff = np.abs(mol_pos_new - mol_pos_old)
    mol_pos_diff = np.linalg.norm(mol_pos_diff, axis=1) 
    if np.max(mol_pos_diff) < pos_conv:
        print('-----')
        break

######################################################################################################################
##########################           VASP Geometry Optimization                  #####################################
######################################################################################################################

# 'Normal' VASP geometry optimization
if do_vasp_opt:
    # Prepare calculator for VASP geometry optimization
    del calculator
    calculator = Vasp(**normal_opt_vasp_settings)
    full.set_calculator(calculator)

    # Get indices of the atoms forming the molecule
    mol_indices = get_mol_indices(full=full, middle_height=middle_height)
    mol_list.append(copy_mol(mol))

    # Save center of mass of current geometry
    mol_com = mol.get_center_of_mass()
    pos_list.append(mol_com)

    # Get inertia matrix 
    mol_inertia, mol_inertia_inv = get_inertia_mat_and_inv(mol=full[mol_indices])
    mol_inertia_list.append(mol_inertia)
    mol_inertia_inv_list.append(mol_inertia_inv)

    # Save rotation angles of current geometry
    rot_list.append(mol.euler_angles)

    # Do optimization and get energy
    e = full.get_potential_energy(force_consistent=True)
    f = full.get_forces(apply_constraint=False)

    # Save energy of current geometry
    energy_list.append(e)

    # Get net force on molecule
    f_center = get_force_mol(mol_indices=mol_indices, f=f)
    f_center_list.append(f_center)

    # get torque on molecule
    t_center = get_torque_mol(full=full, mol_indices=mol_indices, f=f)
    t_center_list.append(t_center)

######################################################################################################################
##########################                 Finish the job                        #####################################
######################################################################################################################

# Export optimization data from rigid and vasp optim
optim_data = {'position': pos_list, 'rotation': rot_list, 'energy': energy_list, 'force': f_center_list, 'torque': t_center_list, 'inertia':mol_inertia_list, 'inertia_inv':mol_inertia_inv_list, 'mol':mol_list}
f = open('optim_data.pk','wb')
pickle.dump(optim_data, f)
f.close()

print("\n")
print('///////////////////////////////////////////////////////')
print('Summary of Optimization:')
print('Energy:')
print(optim_data['energy'])
print('Position:')
print(optim_data['position'])
print('Force:')
print(optim_data['force'])
print('Torque:')
print(optim_data['torque'])
print('Rotation')
print(optim_data['rotation'])

print("\n")
print('Final Geometry:')
print("Energy: "+str(optim_data['energy'][-1]))
print("Position: "+str(optim_data['position'][-1]))
print("Rotation: "+str(optim_data['rotation'][-1]))
print("Force: "+str(optim_data['force'][-1]))
print("Torque: "+str(optim_data['torque'][-1]))

print("\n")
print('Duration / h:')
print((time.time()-t1)/3600)
print('///////////////////////////////////////////////////////')
