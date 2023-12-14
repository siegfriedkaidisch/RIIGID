import numpy as np

from ase import Atom, Atoms

def get_indices_of_atoms1_in_atoms2(atoms1, atoms2, cutoff=1e-4):
        '''
        Find the indices of the (the atoms of) atoms1 inside the atoms2 object
        #additionally returns a bool, specifying if all atoms have been found

        # maybe not here but as misc function in separate file?
        '''
        atomic_indices = []
        for a1 in atoms1:
            for a2 in atoms2:
                if np.linalg.norm(a1.position - a2.position) < cutoff:
                    atomic_indices.append(a2.index)

        if len(atomic_indices) == len(atoms1):
            return atomic_indices, True
        elif len(atomic_indices) < len(atoms1):
            return atomic_indices, False
        else:
            raise Exception('More atoms found than looked for... Are there atoms unphysically close to each other, or duplicate atoms?')
        

def get_mol_indices_old(full, middle_height, above=True):
    '''
    Given the full system ("full"=surface+molecule), find the indices of the molecule's atoms. 
    To do so, specifiy a height ("middle_height") separating the surface from the molecule. 

    above=True: Atoms below middle_height are considered to belong to the surface and atoms above middle_height 
    are considered to belong to the molecule.

    above=False: Atoms above middle_height are considered to belong to the surface and atoms below middle_height 
    are considered to belong to the molecule.

    Inputs:
        full: ase.atoms.Atoms
            The full system (surface+molecule) under study
        middle_height: number
            Height (in Angstroem) used to separate molecule from surface (see description above)
        above: Bool
            See explanation given above

    Returns:
        list of length n_atoms_in_molecule
            List containing indices of the molecule's atoms in "full"
    '''
    if above:
        return [ atom.index for atom in full if atom.position[2] >= middle_height ]
    else:
        return [ atom.index for atom in full if atom.position[2] < middle_height ]
    
def copy_docstring(take_from_fct): 
    docstring = take_from_fct.__doc__
    def decorator(give_to_fct):                                                                                                                                                                                                           
        give_to_fct.__doc__ = docstring   
        return give_to_fct                                                                                                                                                                                            
    return decorator