#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:49:28 2023

Find optimal distance (and angle) between two rigid molecules

@author: kaidisch_siegfried
"""
import numpy as np
import pickle
import time
import os

from ase import Atom, Atoms
from ase.build import add_adsorbate, make_supercell, sort, add_vacuum
from ase.visualize import view
from ase.io import read
from ase.constraints import FixAtoms, FixBondLengths
import os
from ase.optimize import BFGS, MDMin
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG

from ase.io import read, write

from ase.calculators.orca import OrcaProfile
from ase.calculators.orca import ORCA
from pathlib import Path

from lib.rotation_functions import add_rot_props_mol, rotate_mol2, get_inertia_mat_and_inv, get_torque_mol
from lib.translation_functions import get_force_mol
from lib.transrot_functions import random_step_mol, transrot_mol
from lib.boundary_functions import apply_boundaries_mol
from lib.misc_functions import get_mol_indices, copy_mol
    
t1 = time.time()
######################################################################################################################
##########################                      Settings                         #####################################
######################################################################################################################
# Set the working directory
working_dir = '/gpfs/data/fs71335/skaidisch/development/rigid_optim/23.11.14/two_rigid_molecules/'
working_dir = Path(working_dir)

# Geometry Settings
initial_separation = 3 #Angstroem, initial separation between molecules
middle_height      = 0.1 #Angstroem, used to separate molecules

# Settings for rigid geometry optimization
max_rigid_steps = 200 # maximal number of rigid optimization steps
pos_conv = 0.00000000001 # in Angstroem, stop rigid optimization if all atoms of the molecule move less than pos_conv (Warning: may coverge due to small stepsizes rather than actual convergence)
max_bad_steps = 5 # if energy cannot be lowered max_bad_steps steps in a row, stop rigid optimization
max_step_0 = 0.3 #max allowed change of position of an atom (translation+rotation) in iteration 0
max_step = 0.3 #max allowed change of position of an atom (translation+rotation) in iterations 1+
stepsize_factor_up = 1.2 # increase stepsize by this factor, if last iteration successfully lowered energy
stepsize_factor_dn = 0.2 # decrease stepsize by this factor, if last iteration led to higher energy

# Flags to control motion
do_trans_x = 0
do_trans_y = 0
do_trans_z = 1
do_rot     = 0
z_only_rot = False

# Where is your orca executable?
MyOrcaProfile = OrcaProfile("/gpfs/opt/sw/skylake/spack-0.19.0/opt/spack/linux-almalinux8-skylake_avx512/gcc-12.2.0/orca-5.0.4-rmsud3mol2l2zzkksrembfofifwfbcrl/bin/orca")

# Calculator settings for rigid optimization
charge = 0
multiplicity = 1
orcasimpleinput_rigid = 'TightOpt TightSCF B3LYP D3BJ 6-311G*'
orcablocks_rigid = '%geom MaxIter 1 end \n %pal nprocs 48 end \n %output xyzfile true end \n %output print [p_InputFile] 1 end \n %output PrintLevel Normal end'
rigid_opt_dir = working_dir / 'rigid_opt'
rigid_opt_dir.mkdir(parents=True, exist_ok=True)

# Randomness used to escape saddle points (at the moment, only used to escape saddle point initial guesses)
do_random_step      = False
random_seed         = 12345
random_displacement = 0.01 #Angstroem
random_angle        = 0.5 #Degrees

# A list of Orca input and output files
orca_files = [
    "orca.inp",
    "orca_property.txt",
    "orca.densities",
    "orca.gbw",
    "orca.xyz",
    "orca.out",

    "orca.engrad",              
    "orca_trj.xyz",              
    "orca.opt"
    ]            

######################################################################################################################
##########################           Prepare Geometry Optimization               #####################################
######################################################################################################################

# Import geometry of both molecules
mol_base = read(str(working_dir)+'/ptcda.xyz')
mol = read(str(working_dir)+'/h2tpp.xyz')

# Align molecules' center of masses and position mol above mol_base
mol_base.positions -= mol_base.get_center_of_mass()
mol.positions -= mol.get_center_of_mass()
mol_height = max(mol_base.positions[:,2]) + initial_separation
mol.positions[:,2] += - min(mol.positions[:,2]) + mol_height
full = mol_base+mol

# Apply random step to possibly leave a saddle point
mol_indices = get_mol_indices(full=full, middle_height=middle_height, above=True)
del full[mol_indices]
add_rot_props_mol(mol)
if do_random_step:
    # Random step to get rid of symmetry in starting point
    random_step_mol(mol=mol, displacement=random_displacement, angle=random_angle, seed=random_seed)
full += mol.copy()

# Sort atoms
full = sort(full)

# Export geometry for documentation purposes
write(str(working_dir)+'/dimer_start.xyz', full, format='xyz')
#view(full)

######################################################################################################################
##########################          RIGID Geometry Optimization                  #####################################
######################################################################################################################

# Set up Orca calculator 
calc = ORCA(profile=MyOrcaProfile,
            charge=charge, mult=multiplicity,
            directory = working_dir,
            orcasimpleinput=orcasimpleinput_rigid,
            orcablocks=orcablocks_rigid
            )

# Set calculator
full.set_calculator(calc)

# Optimize location and rotation of rigid molecule on rigid surface
stepsize = 100 # the value set here doesn't matter, it's just for the first test step, so it will be changed anyway
count_bad_steps = 0 # needed to cope with bad iterations (where energy got larger)
mol_list = []
mol_separation_list = []
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
    mol_indices = get_mol_indices(full=full, middle_height=middle_height, above=True)
    mol_base_indices = get_mol_indices(full=full, middle_height=middle_height, above=False)

    # Get separation between molecules:
    max_z_mol_base = np.max(full[mol_base_indices].positions[:,2])
    min_z_mol      = np.min(full[mol_indices].positions[:,2])
    mol_separation = np.abs(min_z_mol-max_z_mol_base)
    mol_separation_list.append(mol_separation)

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
    mol_pos_new = transrot_mol(mol=mol_test, f_center=f_center, stepsize_trans_x=stepsize*do_trans_x, stepsize_trans_y=stepsize*do_trans_y, stepsize_trans_z=stepsize*do_trans_z, 
                                        mol_inertia_inv=mol_inertia_inv, t_center=t_center, stepsize_rot=stepsize*do_rot, z_only_rot=z_only_rot)
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
    mol_pos_new = transrot_mol(mol=mol, f_center=f_center, stepsize_trans_x=stepsize*do_trans_x, stepsize_trans_y=stepsize*do_trans_y, stepsize_trans_z=stepsize*do_trans_z, 
                                        mol_inertia_inv=mol_inertia_inv, t_center=t_center, stepsize_rot=stepsize*do_rot, z_only_rot=z_only_rot)

    # Apply boundary conditions
    #apply_boundaries_mol(mol=mol, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)    

    full += mol.copy()

    #####################################################################

    # Print some stuff for testing purposes
    #print('Displacement/stepsize: ' + str(f_center/np.sum(mol.get_masses())))
    #print('Angle/stepsize: ' + str(np.linalg.norm(mol_inertia_inv@t_center) * (180/np.pi)))
    #print('-----')

    # Save geometry
    write(str(rigid_opt_dir)+'/dimer_rigid_iter_' + str(i) + '.xyz', full, format='xyz')

    # Check for convergence
    mol_pos_diff = np.abs(mol_pos_new - mol_pos_old)
    mol_pos_diff = np.linalg.norm(mol_pos_diff, axis=1) 
    if np.max(mol_pos_diff) < pos_conv or count_bad_steps > max_bad_steps:
        print('-----')
        break

######################################################################################################################
##########################                 Finish the job                        #####################################
######################################################################################################################

# Export optimization data from rigid and vasp optim
optim_data = {'position': pos_list, 'rotation': rot_list, 'energy': energy_list, 'force': f_center_list, 'torque': t_center_list, 'inertia':mol_inertia_list, 'inertia_inv':mol_inertia_inv_list, 'mol':mol_list, 'mol_separation':mol_separation_list}
f = open(str(working_dir) + '/optim_data.pk','wb')
pickle.dump(optim_data, f)
f.close()

print("\n")
print('///////////////////////////////////////////////////////')
print('Summary of Optimization:')
print('Energy:')
print(optim_data['energy'])
#print('Position:')
#print(optim_data['position'])
#print('Force:')
#print(optim_data['force'])
#print('Torque:')
#print(optim_data['torque'])
#print('Rotation')
#print(optim_data['rotation'])
print('Separation between molecules')
print(optim_data['mol_separation'])

print("\n")
print('Final Geometry:')
print("Energy: "+str(optim_data['energy'][-1]))
#print("Position: "+str(optim_data['position'][-1]))
#print("Rotation: "+str(optim_data['rotation'][-1]))
#print("Force: "+str(optim_data['force'][-1]))
#print("Torque: "+str(optim_data['torque'][-1]))
print("Separation between molecules: "+str(optim_data['mol_separation'][-1]))

# Move files of the rigid optimization to the subdirectory
for filename in orca_files:
    if os.path.isfile(str(working_dir) + '/' + filename):
        os.rename(str(working_dir) + '/' + filename,str(rigid_opt_dir) + '/' + filename)

print("\n")
print('Duration / h:')
print((time.time()-t1)/3600)
print('///////////////////////////////////////////////////////')
