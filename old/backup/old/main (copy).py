#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:49:28 2023

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

# Randomness used to escape saddle points
do_random_step      = True
random_seed         = 12345
random_displacement = 0.02 #Angstroem
random_angle        = 0.5 #Degrees

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
def angle_between_vectors(v1,v2): 
    '''
    Calculates the angle (in degrees) between two vectors in 3D

    Inputs:
        v1, v2: list of length 3 or numpy.ndarray of shape (3,)
            The two vectors

    Returns:
        number
            The angle (in degrees) between the two vectors
    '''
    return np.arccos(np.dot(v1,v2)/( np.linalg.norm(v1)*np.linalg.norm(v2) ))*180/np.pi

def angles_between_principal_axes_and_xyz(mat_inertia):
    '''
    Takes the inertia tensor of a molecule, calculates the principal axes of inertia (eigenvectors) and then calculates
    the angles (in degrees) between these principal axes and the space-fixed x,y,z- axis. 
    Can be used to identify the current rotation/orientation of the molecule, even in (non-rigid) VASP geometry optimizations.

    Inputs:
        mat_inertia: numpy.ndarray of shape (3,3)
            The inertia tensor of the molecule

    Returns:
        numpy.ndarray of shape (3,3)
            Matrix containing angels (in degrees) between principal axes and the x,y,z- axis; 
            The element [i,j] of this matrix is the angle between principle axis j and axis i (i=0 means x, 1=y, 2=z)
    '''
    eigvals, eigvecs = np.linalg.eig(mat_inertia)
    eigvecs = eigvecs[:,np.argsort(eigvals)] #sort by size of eigenvector
    angles = np.array([[angle_between_vectors(eigvec,axis) for axis in [[1,0,0],[0,1,0],[0,0,1]]] for eigvec in eigvecs])
    return angles

def get_mol_indices(full, middle_height):
    '''
    Given the full system ("full"=surface+molecule), find the indices of the molecule's atoms. 
    To do so, specifiy a height ("middle_height") separating the surface from the molecule. 
    I.e.: Atoms below middle_height are considered to belong to the surface and atoms above middle_height 
    are considered to belong to the molecule.

    Inputs:
        full: ase.atoms.Atoms
            The full system (surface+molecule) under study
        middle_height: number
            Height (in Angstroem) used to separate molecule from surface (see description above)

    Returns:
        list of length n_atoms_in_molecule
            List containing indices of the molecule's atoms in "full"
    '''
    return [ atom.index for atom in full if atom.position[2] > middle_height ]

def get_inertia_mat_and_inv(mol):
    '''
    Get inertia tensor (and its inverse) of a molecule.

    Inputs:
        mol: ase.atoms.Atoms
            The molecule whose inertia matrix (and inverse) will be calculated

    Returns:
        numpy.ndarray of shape (3,3)
            The inertia matrix of the molecule (in Dalton*Angstroem**2)
        numpy.ndarray of shape (3,3)
            The inverse inertia matrix of the molecule (in 1/(Dalton*Angstroem**2))
    '''
    mol_com = mol.get_center_of_mass()
    mol_inertia = np.zeros([3,3])
    for j in range(3):
        for k in range(3):
            for atom in mol:
                r_l = atom.position - mol_com
                if j==k:
                    mol_inertia[j,k] += atom.mass * ( np.linalg.norm(r_l)**2 - r_l[j]*r_l[k] ) 
                else:
                    mol_inertia[j,k] += atom.mass * (                - r_l[j]*r_l[k] ) 
    mol_inertia_inv = np.linalg.inv(mol_inertia)
    return mol_inertia, mol_inertia_inv

def get_force_mol(mol_indices, f):
    '''
    Get the net force acting on the molecule.

    Inputs:
        mol_indices: list of length n_atoms_in_molecule
            List containing indices of the molecule's atoms in "full"
        f: numpy.ndarray of shape (n_atoms_in_full_system, 3)
            Forces acting on the atoms in "full" (in eV/Angstroem)

    Returns:
        numpy.ndarray of shape (3,)
            Net force acting on the molecule (in eV/Angstroem)
    '''
    mol_f = f[mol_indices]
    f_center = np.sum(mol_f, axis=0)
    return f_center

def get_torque_mol(full, mol_indices, f):
    '''
    Get the net torque acting on the molecule (relative to its center of mass).

    Inputs:
        full: ase.atoms.Atoms
            The full system (surface+molecule) under study
        mol_indices: list of length n_atoms_in_molecule
            List containing indices of the molecule's atoms in "full"
        f: numpy.ndarray of shape (n_atoms_in_full_system, 3)
            Forces acting on the atoms in "full" (in eV/Angstroem)

    Returns:
        numpy.ndarray of shape (3,)
            Net torque acting on the molecule (relative to center of mass of molecule) (in eV)
    '''
    mol_com = full[mol_indices].get_center_of_mass()
    t_center = np.array([0.0,0.0,0.0])
    for index in mol_indices:
        r_i = full[index].position
        r   = mol_com
        f_i = f[index]
        t_center += np.cross(r_i-r, f_i)
    return t_center

def translate_mol(mol, f_center, stepsize_trans_x, stepsize_trans_y, stepsize_trans_z):
    '''
    Translate molecule in direction of the applied force

    Inputs:
        mol: ase.atoms.Atoms
            The molecule to be translated
        f_center: numpy.ndarray of shape (3,) or list of length 3
            Net force acting on the molecule (in eV/Angstroem)
        stepsize_trans_x/y/z: number
            Timesteps; usually all three should have the same value; (in Dalton*Angstroem**2/eV)

    Returns:
        numpy.ndarray of shape (n_atoms_in_molecule,3)
            The positions (in Angstroem) of the molecule's atoms after the transformation
    '''
    mol_mass = np.sum(mol.get_masses())
    for atom in mol:
        atom.position[0] += stepsize_trans_x * f_center[0]/mol_mass
        atom.position[1] += stepsize_trans_y * f_center[1]/mol_mass
        atom.position[2] += stepsize_trans_z * f_center[2]/mol_mass
    return mol.positions.copy()

def rotate_mol(mol, mol_inertia_inv, t_center, stepsize_rot):
    '''
    Rotate molecule around its center of mass following the applied torque

    Inputs:
        mol: ase.atoms.Atoms
            The molecule to be rotated
        mol_inertia_inv: numpy.ndarray of shape (3,3)
            The inverse inertia matrix of the molecule (in 1/(Dalton*Angstroem**2))
        t_center: numpy.ndarray of shape (3,) or list of length 3
            Net torque acting on the molecule (relative to center of mass of molecule) (in eV)
        stepsize_rot: number
            Timestep (in Dalton*Angstroem**2/eV)

    Returns:
        numpy.ndarray of shape (n_atoms_in_molecule,3)
            The positions (in Angstroem) of the molecule's atoms after the transformation
    '''
    t_center = np.array(t_center)
    mol_com = mol.get_center_of_mass()
    tmp = mol_inertia_inv@t_center
    angle = np.linalg.norm(tmp) * (180/np.pi) * stepsize_rot  # in degrees
    axis = tmp/np.linalg.norm(tmp)
    #angle = tmp[2] * stepsize_rot 
    #axis = 'z'
    center = mol_com
    mol.rotate(angle,axis,center)
    return mol.positions.copy()

def transrot_mol(mol, f_center, stepsize_trans_x, stepsize_trans_y, stepsize_trans_z, 
                    mol_inertia_inv, t_center, stepsize_rot):
    '''
    Translate molecule in direction of the applied force and rotate it around its center of mass following the applied torque

    Inputs:
        mol: ase.atoms.Atoms
            The molecule to be translated
        f_center: numpy.ndarray of shape (3,) or list of length 3
            Net force acting on the molecule (in eV/Angstroem)
        stepsize_trans_x/y/z/rot: number
            Timesteps; usually all four should have the same value; (in Dalton*Angstroem**2/eV)
        mol_inertia_inv: numpy.ndarray of shape (3,3)
            The inverse inertia matrix of the molecule (in 1/(Dalton*Angstroem**2))
        t_center: numpy.ndarray of shape (3,) or list of length 3
            Net torque acting on the molecule (relative to center of mass of molecule) (in eV)

    Returns:
        numpy.ndarray of shape (n_atoms_in_molecule,3)
            The positions (in Angstroem) of the molecule's atoms after the transformation
    '''
    # Translate molecule
    translate_mol(mol=mol, f_center=f_center, stepsize_trans_x=stepsize_trans_x, stepsize_trans_y=stepsize_trans_y, stepsize_trans_z=stepsize_trans_z)
    # Rotate molecule 
    rotate_mol(mol=mol, mol_inertia_inv=mol_inertia_inv, t_center=t_center, stepsize_rot=stepsize_rot)
    return mol.positions.copy()

def random_step_mol(mol, displacement, angle, seed):
    '''

    '''
    stepsize_trans_x = 1
    stepsize_trans_y = 1
    stepsize_trans_z = 1
    stepsize_rot     = 1
    mol_inertia_inv  = np.eye(3)
    mol_mass = np.sum(mol.get_masses())
    
    backup_seed = np.random.randint(2**32 - 1)
    np.random.seed(seed)
    f_center = np.random.rand(3)
    t_center = np.random.rand(3) 
    np.random.seed(backup_seed)
    
    f_center /= np.linalg.norm(f_center)
    t_center /= np.linalg.norm(t_center)
    f_center *= displacement * mol_mass
    t_center *= angle * (np.pi/180)
    
    # Translate molecule
    translate_mol(mol=mol, f_center=f_center, stepsize_trans_x=stepsize_trans_x, stepsize_trans_y=stepsize_trans_y, stepsize_trans_z=stepsize_trans_z)
    # Rotate molecule 
    rotate_mol(mol=mol, mol_inertia_inv=mol_inertia_inv, t_center=t_center, stepsize_rot=stepsize_rot)
    return mol.positions.copy()

def get_normal_vector_mol(mol, lower, upper):
    '''
    Return vector normal (positive z) to planar molecule
    '''
    center = mol.get_center_of_mass()
    normal_vector = np.zeros(3)
    atom1 = mol[0]
    for atom2 in mol:
        r1 = atom1.position - center
        r2 = atom2.position - center
        angle = angle_between_vectors(v1=r1,v2=r2)
        if angle > lower and angle < upper:
            normal_vector = np.cross(r1,r2)
            if normal_vector[2] < 0:
                normal_vector = np.cross(r2,r1)
            break
    normal_vector /= np.linalg.norm(normal_vector)
    return normal_vector

######################################################################################################################
t1 = time.time()

# Import Geometry
full = read_vasp(file='./POSCAR_input')

# Control initial geometry
mol_indices = get_mol_indices(full=full, middle_height=middle_height)
mol = full[mol_indices].copy()
del full[mol_indices]
mol.positions[:,0] += -mol.get_center_of_mass()[0] + pos_x
mol.positions[:,1] += -mol.get_center_of_mass()[1] + pos_y
mol.rotate(rot_z,'z', mol.get_center_of_mass())
if do_random_step:
    # Random step to get rid of symmetry in starting point
    random_step_mol(mol=mol, displacement=random_displacement, angle=random_angle, seed=random_seed)
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
    mol_list.append(mol.copy())

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
    rot_list.append(angles_between_principal_axes_and_xyz(mol_inertia))
    
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
            mol = mol_list[-1 - count_bad_steps].copy()
            f_center = f_center_list[-1 - count_bad_steps]
            mol_inertia_inv = mol_inertia_inv_list[-1 - count_bad_steps]
            t_center = t_center_list[-1 - count_bad_steps]
    print(stepsize)

    # Prepare movement of molecule
    del full[mol_indices]
    mol_pos_old = mol.positions.copy()

    # Test step to find max displacement of atoms
    mol_test = mol.copy()
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
    print(stepsize)

    # Move molecule and get new positions
    mol_pos_new = transrot_mol(mol=mol, f_center=f_center, stepsize_trans_x=stepsize, stepsize_trans_y=stepsize, stepsize_trans_z=stepsize, 
                                        mol_inertia_inv=mol_inertia_inv, t_center=t_center, stepsize_rot=stepsize)

    full += mol.copy()

    #####################################################################

    # Print some stuff for testing purposes
    print(f_center/np.sum(mol.get_masses()))
    print(np.linalg.norm(mol_inertia_inv@t_center) * (180/np.pi))
    print('-----')

    # Save geometry
    write_vasp('./rigid_opt/CONTCAR_'+str(i), full, direct=True, vasp5=True, sort=True)
    
    # Check for convergence
    mol_pos_diff = np.abs(mol_pos_new - mol_pos_old)
    mol_pos_diff = np.linalg.norm(mol_pos_diff, axis=1) 
    if np.max(mol_pos_diff) < pos_conv:
        break


######################################################################################################################

# 'Normal' VASP geometry optimization
if do_vasp_opt:
    # Prepare calculator for VASP geometry optimization
    del calculator
    calculator = Vasp(**normal_opt_vasp_settings)
    full.set_calculator(calculator)

    # Get indices of the atoms forming the molecule
    mol_indices = get_mol_indices(full=full, middle_height=middle_height)
    mol_list.append(mol.copy())

    # Save center of mass of current geometry
    mol_com = mol.get_center_of_mass()
    pos_list.append(mol_com)

    # Get inertia matrix 
    mol_inertia, mol_inertia_inv = get_inertia_mat_and_inv(mol=full[mol_indices])
    mol_inertia_list.append(mol_inertia)
    mol_inertia_inv_list.append(mol_inertia_inv)

    # Save rotation angles of current geometry
    rot_list.append(angles_between_principal_axes_and_xyz(mol_inertia))

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

# Export optimization data from rigid and vasp optim
optim_data = {'position': pos_list, 'rotation': rot_list, 'energy': energy_list, 'force': f_center_list, 'torque': t_center_list, 'inertia':mol_inertia_list, 'inertia_inv':mol_inertia_inv_list, 'mol':mol_list}
f = open('optim_data.pk','wb')
pickle.dump(optim_data, f)
f.close()
print('///////////////////////////////////////////////////////')
print('')
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

print('Duration / h:')
print((time.time()-t1)/3600)
print('///////////////////////////////////////////////////////')
