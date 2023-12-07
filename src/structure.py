from copy import deepcopy
import numpy as np

from fragment import Fragment

class Structure():
    def __init__(self, atoms):
        self.atoms = atoms
        self.fragments = []
        self.rest_fragment = Fragment(atoms=self.atoms, translation='', rotation='')

    def define_fragment_by_indices(self, indices, translation, rotation):
        '''
        Takes atoms from rest_fragment to from new fragment (indices relative to full atoms object)
        '''
        fragment_atoms = deepcopy(self.atoms[indices])
        rest_fragment_atoms = deepcopy(self.rest_fragment.atoms)

        new_fragment = Fragment(atoms=fragment_atoms, translation=translation, rotation=rotation)

        # Remove atoms of this new fragment from the rest-fragment
        indices_of_new_fragment_in_rest_fragment, found = self.get_indices_of_atoms1_in_atoms2(atoms1=fragment_atoms, atoms2=fragment_atoms)
        if not found:
            raise Exception('Atoms not found in rest-fragment. Did you already include some of the atoms in another fragment? Every atom has to belong to exactly one fragment. (All atoms not assigned to any fragment form the rest-fragment.)')
        del rest_fragment_atoms[indices_of_new_fragment_in_rest_fragment]

        # Update rest-fragment and append new fragment to list of fragments
        self.rest_fragment = Fragment(atoms=rest_fragment_atoms, translation='', rotation='')
        self.fragments.append(new_fragment)

    def get_indices_of_atoms1_in_atoms2(atoms1, atoms2, cutoff=1e-4):
            '''
            Find the indices of the (the atoms of) atoms1 inside the atoms2 object
            #additionally returns a bool, specifying if all atoms have been found
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

    def calculate_energy_and_forces(self, calculator):
        atoms = deepcopy(self.atoms)
        atoms.set_calculator(calculator)
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        return energy, forces
    
    def calculate_forces_on_fragments(self, forces):
        forces_on_fragments = []
        for fragment in self.fragments:
            fragment_indices, found = self.get_indices_of_atoms1_in_atoms2(atoms1=fragment.atoms, atoms2=self.atoms, cutoff=1e-4)



    def move(self, forces, stepsize, change):
        #calculate forces on fragments

        for i, fragment in enumerate(self.framgnets):
            fragmnet.move(forces[self.indcies[i]])


        #update self.atoms by summing up all fragments.atoms (separate method for this update)

