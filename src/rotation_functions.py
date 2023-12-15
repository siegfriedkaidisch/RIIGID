"""
A collection of functions related to the rotation of fragments.

They are stored here, in a separate file, because they may also be useful somewhere else than just 
inside the Fragment class.

"""
import numpy as np

######################################################################################################################
# The following functions are currently used by RIGID.

def angle_between_vectors(v1,v2): 
    """Calculates the angle (in degrees) between two vectors in 3D.

    Parameters
    ----------
    v1, v2: list of length 3 or numpy.ndarray of shape (3,)
        The two vectors

    Returns
    -------
    number between 0 and 180
        The angle (in degrees) between the two vectors

    """
    return np.arccos(np.dot(v1,v2)/( np.linalg.norm(v1)*np.linalg.norm(v2) ))*180/np.pi

def signed_angle_between_vectors(v1,v2, axis): 
    """Calculates the "signed" angle (in degrees) between two vectors in 3D. 

    v1 and v2 have to be normal to axis. The angle is right-hand measured around axis, from v1 to v2
    
    Parameters
    ----------
    v1, v2, axis: list of length 3 or numpy.ndarray of shape (3,)
        The two vectors and the axis

    Returns
    -------
    number between 0 and 360
        The angle (in degrees) between the two vectors

    """
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
    """Create 3x3 rotation matrix for the rotation around axis by angle.

    Parameters
    ----------
    axis: list of length 3 or numpy.ndarray of shape (3,)
        The rotation axis
    angle: number 
        The rotation angle
    
    Returns
    -------
    np.ndarray of shape (3,3)
        The rotation matrix

    """
    angle *= np.pi/180 #convert to radians
    axis = np.array(axis)
    axis/= np.linalg.norm(axis)
    
    unit_mat       = np.eye(3)
    cross_prod_mat = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    outer_prod_mat = np.outer(axis,axis) 
    rot_mat = np.cos(angle)*unit_mat + np.sin(angle)*cross_prod_mat + (1-np.cos(angle))*outer_prod_mat
    return rot_mat

######################################################################################################################
# The following functions are currently not used by RIGID
# They are still stored here, for possible future purposes

def angles_between_principal_axes_and_xyz(mat_inertia):
    """Calculate angles between x,y,z-axis and principal axes of inertia.

    Takes the inertia matrix of a fragment, calculates the principal axes of inertia (eigenvectors) and then calculates
    the angles (in degrees) between these principal axes and the space-fixed x,y,z- axis. 
    Can be used to identify the current rotation/orientation of the fragment, even in non-rigid (e.g. VASP) geometry optimizations.

    Parameters
    ----------
    mat_inertia: np.ndarray of shape (3,3)
        The inertia matrix of the fragment

    Returns
    -------
    np.ndarray of shape (3,3)
        Matrix containing angels (in degrees) between principal axes and the x,y,z- axis; 
        The element [i,j] of this matrix is the angle between principle axis j and axis i (i=0 means x, 1=y, 2=z)
    
    """
    eigvals, eigvecs = np.linalg.eig(mat_inertia)
    eigvecs = eigvecs[:,np.argsort(eigvals)] #sort by size of eigenvector
    angles = np.array([[angle_between_vectors(eigvec,axis) for axis in [[1,0,0],[0,1,0],[0,0,1]]] for eigvec in eigvecs])
    return angles

def get_normal_vector_fragment(fragment):
    """Find and return the (normalized) vector normal to a planar (non-linear) fragment.

    The normal vector is calculated by looking for two atoms that form a near 90 degree angle 
    (for numeric stability) and then calculating the normalized cross-product of these two vectors. 
    The resulting vector is normal to the planar molecule.

    The normal vector is obviously not unique, there are two possible directions. This function uses 
    the following convention:
    If the normal vector has a nonzero z-component, then choose it to be positive. 
    Otherwise, if it has a nonzero y-component, then choose this y-component to be positive. 
    Otherwise choose the x-component to be positive.

    Note: DO NOT USE THIS FUNCTION WITH A NON-PLANAR OR A LINEAR FRAGMENT!

    Parameters
    ----------
    fragment: fragment.Fragment
        The planar, non-linear fragment whose normal vector shall be calculated

    Returns
    -------
    np.ndarray of shape (3,)
        The normal vector

    """
    center = fragment.atoms.get_center_of_mass()
    normal_vector = np.zeros(3)
    best_found_angle = 0 # the closer to 90 degrees, the better
    for atom1 in fragment.atoms:
        for atom2 in fragment.atoms:
            if atom1 != atom2:
                r1 = atom1.position - center
                r2 = atom2.position - center
                angle = angle_between_vectors(v1=r1,v2=r2)
                if np.abs(angle-90) < np.abs(best_found_angle-90):
                    best_found_angle = angle
                    normal_vector = np.cross(r1,r2)

    if normal_vector == np.zeros(3):
        raise Exception("Something went wrong... normal vector could not be determined.")
    else: # Apply convention for direction
        if normal_vector[2] == 0:
            if normal_vector[1] == 0:
                if normal_vector[0] < 0:
                    normal_vector *= -1
            elif normal_vector[1] < 0:
                normal_vector *= -1
        elif normal_vector[2] < 0:
            normal_vector *= -1
    normal_vector /= np.linalg.norm(normal_vector)
    return normal_vector

######################################################################################################################
