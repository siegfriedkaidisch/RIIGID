from copy import deepcopy
import numpy as np

from fragment import Fragment
from misc_functions import get_indices_of_atoms1_in_atoms2

class Structure():
    def __init__(self, atoms):
        self.atoms = atoms
        self.fragments = []
        self.rest_fragment = Fragment(atoms=self.atoms, allowed_translation='', allowed_rotation='')

    def define_fragment_by_indices(self, indices, allowed_translation, allowed_rotation):
        '''
        Takes atoms from rest_fragment to from new fragment (indices relative to full atoms object)
        '''
        fragment_atoms = deepcopy(self.atoms[indices])
        rest_fragment_atoms = deepcopy(self.rest_fragment.atoms)

        new_fragment = Fragment(atoms=fragment_atoms, allowed_translation=allowed_translation, allowed_rotation=allowed_rotation)

        # Remove atoms of this new fragment from the rest-fragment
        indices_of_new_fragment_in_rest_fragment, found = get_indices_of_atoms1_in_atoms2(atoms1=fragment_atoms, atoms2=fragment_atoms)
        if not found:
            raise Exception('Atoms not found in rest-fragment. Did you already include some of the atoms in another fragment? Every atom has to belong to exactly one fragment. (All atoms not assigned to any fragment form the rest-fragment.)')
        del rest_fragment_atoms[indices_of_new_fragment_in_rest_fragment]

        # Update rest-fragment and append new fragment to list of fragments
        self.rest_fragment = Fragment(atoms=rest_fragment_atoms, allowed_translation='', allowed_rotation='')
        self.fragments.append(new_fragment)

    def calculate_energy_and_forces(self, calculator):
        atoms = deepcopy(self.atoms)
        atoms.set_calculator(calculator)
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        return energy, forces
    
    def get_indices_of_fragments(self):
        fragments_indices = []
        for fragment in self.fragments:
            fragment_indices, found = get_indices_of_atoms1_in_atoms2(atoms1=fragment.atoms, atoms2=self.atoms, cutoff=1e-4)
            if not found:
                raise Exception('Error while looking for indices of fragment...')
            fragments_indices.append(fragment_indices)
        return fragments_indices
    
    def get_forces_on_fragments(self, forces):
        fragments_indices = self.get_indices_of_fragments()
        forces_on_fragments = [forces[indices_i] for indices_i in fragments_indices]
        return forces_on_fragments

    def calculate_net_force_on_fragments(self, forces):
        forces_on_fragments = self.get_forces_on_fragments(forces=forces)
        net_force_on_fragments = []
        for i, fragment in enumerate(self.fragments):
            net_force_on_fragment = fragment.calculate_net_force_on_fragment(forces = forces_on_fragments[i])
            net_force_on_fragments.append(net_force_on_fragment)
        return net_force_on_fragments
    
    def calculate_torque_on_fragments(self, forces):
        forces_on_fragments = self.get_forces_on_fragments(forces=forces)
        torque_on_fragments = []
        for i, fragment in enumerate(self.fragments):
            torque_on_fragment = fragment.calculate_torque_on_fragment(forces = forces_on_fragments[i])
            torque_on_fragments.append(torque_on_fragment)
        return torque_on_fragments
    
    def update_atoms_attribute_from_fragments(self):
        self.atoms = self.rest_fragment.atoms + sum([fragment.atoms for fragment in self.fragments])

    def move(self, forces, stepsize):
        old_positions = deepcopy(self.atoms.positions)

        force_on_fragments = self.calculate_force_on_fragments(forces=forces)
        torque_on_fragments = self.calculate_torque_on_fragments(forces=forces)

        for i, fragment in enumerate(self.fragments):
            fragment.move(force_on_fragment=force_on_fragments[i], torque_on_fragment=torque_on_fragments[i], stepsize=stepsize)

        #update self.atoms by summing up all fragments.atoms
        self.update_atoms_attribute_from_fragments()

        new_positions = deepcopy(self.atoms.positions)

        atomic_displacements = new_positions-old_positions
        max_atomic_displacement = np.max(np.linalg.norm(atomic_displacements, axis=1))
        return max_atomic_displacement, atomic_displacements
    