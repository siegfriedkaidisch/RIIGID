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

######################################################################################################################
