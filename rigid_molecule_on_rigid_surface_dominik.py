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
middle_height = 8.5 #in Angstroem, used to separate molecule and surface
vacuum = 10 # in Angstroem, total height of vacuum

# Settings for rigid geometry optimization
max_rigid_steps = 3 # maximal number of rigid optimization steps
pos_conv = 0.00000000001 # in Angstroem, stop rigid optimization if all atoms of the molecule move less than pos_conv (Warning: may coverge due to small stepsizes rather than actual convergence)
max_step_0 = 0.1 #max allowed change of position of an atom (translation+rotation) in iteration 0
max_step = 0.1 #max allowed change of position of an atom (translation+rotation) in iterations 1+
stepsize_factor_up = 1.2 # increase stepsize by this factor, if last iteration successfully lowered energy
stepsize_factor_dn = 0.2 # decrease stepsize by this factor, if last iteration led to higher energy

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
    'encut':300,
    'ediff':1e-5,
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

    'kpts':[1, 1, 1],
    'gamma':True,  # Gamma-centered k-mesh

    'setups':"recommended"
}

######################################################################################################################
##########################           Prepare Geometry Optimization               #####################################
######################################################################################################################

def get_mol_indices2(full, mol, cutoff=1e-4):
    mol_indices = []
    for mol_atom in mol:
        for full_atom in full:
            if np.linalg.norm(mol_atom.position - full_atom.position) < cutoff:
                mol_indices.append(full_atom.index)
    return mol_indices

# Import Geometry
full = read_vasp(file='./POSCAR_Dominik')
mol0 = read_vasp(file='./POSCAR_Dominik_mol0')
mol1 = read_vasp(file='./POSCAR_Dominik_mol1')
mol2 = read_vasp(file='./POSCAR_Dominik_mol2')
mol3 = read_vasp(file='./POSCAR_Dominik_mol3')
molecules = [mol0,mol1,mol2,mol3]
for mol in molecules:
    add_rot_props_mol(mol)

# Control initial geometry
#mol_indices = get_mol_indices(full=full[~(full.symbols=='Ni')], middle_height=middle_height)
mol_indices = [get_mol_indices2(full=full, mol=mol) for mol in molecules]
del full[mol_indices]
full += mol0.copy()
full += mol1.copy()
full += mol2.copy()
full += mol3.copy()

# Sort atoms
full = sort(full)

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
    mol_list.append([copy_mol(mol) for mol in molecules])

    # Get indices of the atoms forming the molecule
    mol_indices = [get_mol_indices2(full=full, mol=mol) for mol in molecules]

    # Save center of mass of current geometry
    mol_com = [mol.get_center_of_mass() for mol in molecules]
    pos_list.append(mol_com)

    # Get inertia matrix and inverse
    tmp = np.array([get_inertia_mat_and_inv(mol=full[mol_indices_i]) for mol_indices_i in mol_indices])
    mol_inertia = tmp[:,0].copy()
    mol_inertia_inv = tmp[:,1].copy()
    del tmp
    mol_inertia_list.append(mol_inertia)
    mol_inertia_inv_list.append(mol_inertia_inv)

    # Save rotation angles of current geometry
    rot_list.append([mol.euler_angles for mol in molecules])
    
    # Run one SCF cycle
    f = full.get_forces(apply_constraint=False)
    e = full.get_potential_energy(force_consistent=True)

    # Save energy of current geometry
    energy_list.append(e)

    # Get net force on molecule
    f_center = [get_force_mol(mol_indices=mol_indices_i, f=f) for mol_indices_i in mol_indices]
    f_center_list.append(f_center)

    # get torque on molecule
    t_center = [get_torque_mol(full=full, mol_indices=mol_indices_i, f=f) for mol_indices_i in mol_indices]
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
            molecules = [copy_mol(mol) for mol in mol_list[-1 - count_bad_steps]]
            f_center = f_center_list[-1 - count_bad_steps]
            mol_inertia_inv = mol_inertia_inv_list[-1 - count_bad_steps]
            t_center = t_center_list[-1 - count_bad_steps]
    print('Test Stepsize: ' + str(stepsize))

    # Prepare movement of molecule
    del full[mol_indices]
    mol_pos_old = [mol.positions.copy() for mol in molecules]

    # Test step to find max displacement of atoms
    molecules_test = [copy_mol(mol) for mol in molecules]
    mol_pos_diff_max = -1
    for i in range(len(molecules)):
        mol_pos_new = transrot_mol(mol=molecules_test[i], f_center=f_center[i], stepsize_trans_x=stepsize, stepsize_trans_y=stepsize, stepsize_trans_z=stepsize, 
                                            mol_inertia_inv=mol_inertia_inv[i], t_center=t_center[i], stepsize_rot=stepsize)
        mol_pos_diff = np.abs(mol_pos_new - mol_pos_old[i])
        mol_pos_diff_max = np.max([np.max(np.linalg.norm(mol_pos_diff, axis=1)), mol_pos_diff,max])
    del molecules_test

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
    mol_pos_new = []
    for i in range(len(molecules)):
        mol_pos_new_i = transrot_mol(mol=molecules[i], f_center=f_center[i], stepsize_trans_x=stepsize, stepsize_trans_y=stepsize, stepsize_trans_z=stepsize, 
                                        mol_inertia_inv=mol_inertia_inv[i], t_center=t_center[i], stepsize_rot=stepsize)
        mol_pos_new.append(mol_pos_new_i)

    full += [mol.copy() for mol in molecules]

    #####################################################################

    # Print some stuff for testing purposes
    #print('Displacement/stepsize: ' + str(f_center/np.sum(mol.get_masses())))
    #print('Angle/stepsize: ' + str(np.linalg.norm(mol_inertia_inv@t_center) * (180/np.pi)))
    #print('-----')

    # Save geometry
    write_vasp('./rigid_opt/CONTCAR_'+str(i), full, direct=True, vasp5=True, sort=True)
    
    # Check for convergence
    mol_pos_diff = np.abs(np.array(mol_pos_new) - np.array(mol_pos_old))
    mol_pos_diff = np.linalg.norm(mol_pos_diff, axis=2) 
    if np.max(mol_pos_diff) < pos_conv:
        print('-----')
        break


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
