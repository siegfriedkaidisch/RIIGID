#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:49:28 2023

@author: kaidisch_siegfried
"""
import numpy as np

from ase import Atom, Atoms

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

def angle_between_vectors2(v1,v2, axis): 
    '''
    v1 and v2 are normal to axis. angle is right-hand measured around axis, from v1 to v2
    
    Calculates the angle (in degrees) between two vectors in 3D

    Inputs:
        v1, v2: list of length 3 or numpy.ndarray of shape (3,)
            The two vectors

    Returns:
        number
            The angle (in degrees) between the two vectors
    '''
    phi = np.arccos(np.dot(v1,v2)/( np.linalg.norm(v1)*np.linalg.norm(v2) ))
    
    sign = np.sign(np.dot( np.cross(v1,v2), axis ))
    if sign == 1:
        None
    else:
        phi = np.pi + np.abs(np.pi-phi)
    if phi==2*np.pi:    
        phi = 0
    return phi * 180/np.pi

def rotmat(axis, angle):
    '''
    angle in degrees
    '''
    angle *= np.pi/180 #convert to radians
    axis = np.array(axis)
    axis/= np.linalg.norm(axis)
    
    unit_mat       = np.eye(3)
    cross_prod_mat = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    outer_prod_mat = np.outer(axis,axis) 
    rot_mat = np.cos(angle)*unit_mat + np.sin(angle)*cross_prod_mat + (1-np.cos(angle))*outer_prod_mat
    return rot_mat













######################################################################################################################

def get_euler_angles(xb,yb,zb, xs=[1,0,0],ys=[0,1,0],zs=[0,0,1]):
    '''
    body-fixed vs space-fixed axis system (body-axes given in space-coords)
    convention: if z axes are parallel, set xs=N
    '''
    xb = np.array(xb)
    yb = np.array(yb)
    zb = np.array(zb)
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    
    beta  = angle_between_vectors(v1=zs, v2=zb)
    
    if beta == 0 or beta==180:
        alpha = 0.0
        gamma = angle_between_vectors2(v1=xs,v2=xb, axis=zb)
    else:
        N = np.cross(zs, zb)
        alpha = angle_between_vectors2(v1=xs,v2=N, axis=zs)
        gamma = angle_between_vectors2(v1=N,v2=xb, axis=zb)
    
    return [alpha, beta, gamma]

def add_rot_props_mol(mol):
    '''
    Add axis system and euler angles to molecule
    '''
    mol.xb = np.array([1.0,0.0,0.0])
    mol.yb = np.array([0.0,1.0,0.0])
    mol.zb = np.array([0.0,0.0,1.0])
    mol.euler_angles = [0.0,0.0,0.0]
    return None

def update_rot_props_mol(mol, angle, axis):
    mat = rotmat(axis=axis,angle=angle)
    mol.xb = mat@mol.xb
    mol.yb = mat@mol.yb
    mol.zb = mat@mol.zb
    mol.euler_angles = get_euler_angles(xb=mol.xb,yb=mol.yb,zb=mol.zb)
    return None

######################################################################################################################

def get_inertia_mat_and_inv(mol):
    '''
    Get inertia tensor (and its inverse) of a molecule.

    Inputs:
        mol: ase.atoms.Atoms
            The molecule whose inertia matrix (and inverse) will be calculated

    Returns:
        np.ndarray of shape (3,3)
            The inertia matrix of the molecule (in Dalton*Angstroem**2)
        np.ndarray of shape (3,3)
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










def angles_between_principal_axes_and_xyz(mat_inertia):
    '''
    Takes the inertia tensor of a molecule, calculates the principal axes of inertia (eigenvectors) and then calculates
    the angles (in degrees) between these principal axes and the space-fixed x,y,z- axis. 
    Can be used to identify the current rotation/orientation of the molecule, even in (non-rigid) VASP geometry optimizations.

    Inputs:
        mat_inertia: np.ndarray of shape (3,3)
            The inertia tensor of the molecule

    Returns:
        np.ndarray of shape (3,3)
            Matrix containing angels (in degrees) between principal axes and the x,y,z- axis; 
            The element [i,j] of this matrix is the angle between principle axis j and axis i (i=0 means x, 1=y, 2=z)
    '''
    eigvals, eigvecs = np.linalg.eig(mat_inertia)
    eigvecs = eigvecs[:,np.argsort(eigvals)] #sort by size of eigenvector
    angles = np.array([[angle_between_vectors(eigvec,axis) for axis in [[1,0,0],[0,1,0],[0,0,1]]] for eigvec in eigvecs])
    return angles

######################################################################################################################

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










def get_torque_mol(full, mol_indices, f):
    '''
    Get the net torque acting on the molecule (relative to its center of mass).

    Inputs:
        full: ase.atoms.Atoms
            The full system (surface+molecule) under study
        mol_indices: list of length n_atoms_in_molecule
            List containing indices of the molecule's atoms in "full"
        f: np.ndarray of shape (n_atoms_in_full_system, 3)
            Forces acting on the atoms in "full" (in eV/Angstroem)

    Returns:
        np.ndarray of shape (3,)
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

######################################################################################################################

def rotate_mol(mol, mol_inertia_inv, t_center, stepsize_rot, z_only_rot=False):
    '''
    Rotate molecule around its center of mass following the applied torque

    Inputs:
        mol: ase.atoms.Atoms
            The molecule to be rotated
        mol_inertia_inv: np.ndarray of shape (3,3)
            The inverse inertia matrix of the molecule (in 1/(Dalton*Angstroem**2))
        t_center: np.ndarray of shape (3,) or list of length 3
            Net torque acting on the molecule (relative to center of mass of molecule) (in eV)
        stepsize_rot: number
            Timestep (in Dalton*Angstroem**2/eV)
        z_only_rot: Bool
            Rotate only around the z-axis?

    Returns:
        np.ndarray of shape (n_atoms_in_molecule,3)
            The positions (in Angstroem) of the molecule's atoms after the transformation
    '''
    t_center = np.array(t_center)
    mol_com = mol.get_center_of_mass()
    tmp = mol_inertia_inv@t_center
    if z_only_rot:
        angle = tmp[2] * (180/np.pi) * stepsize_rot 
    else:
        angle = np.linalg.norm(tmp) * (180/np.pi) * stepsize_rot  # in degrees
    if angle != 0:
        if z_only_rot:
            axis = np.array([0.0,0.0,1.0])
        else:
            axis = tmp/np.linalg.norm(tmp)
        center = mol_com
        mol.rotate(angle,axis,center)
        update_rot_props_mol(mol=mol, angle=angle, axis=axis)
    return mol.positions.copy()

def rotate_mol2(mol, angle, axis):
    '''
    Rotate mol around com and rotate its axis system and update euler angle
    '''   
    mol.rotate(angle,axis, mol.get_center_of_mass())
    update_rot_props_mol(mol=mol, angle=angle, axis=axis)
    return None

######################################################################################################################
