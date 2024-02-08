"""A collection of miscellaneous functions.

"""
import sys
from functools import wraps

import numpy as np

######################################################################################################################
# The following functions are currently used by RIIGID.


def get_indices_of_atoms1_in_atoms2(atoms1, atoms2, cutoff=1e-4):
    """Find the indices of the (the atoms of) atoms1 inside the atoms2 object.

    In the typical use of this function, atoms1 are the atoms of a fragment, while
    atoms2 are the atoms of the full structure.

    To determine the indices, atoms are considered identical, if they are of the same species
    and the distance between them is less than cutoff.

    Parameters
    ----------
    atoms1: ase.atoms.Atoms
        The atoms, whose indices are searched
    atoms2: ase.atoms.Atoms
        The atoms, where atoms1 is searched in
    cutoff: number, default: 1e-4
        Atoms closer than cutoff, which are of the same species, are considered identical

    Returns
    -------
    list of int
        The list of indices; Normally this should be of the same length as atoms1
    bool
        Was the search successful?
        If not all atoms were found (the returned list is shorter than len(atoms1)), this
        is set to False (In this case, the returned list is useless, but still returned for
        compatibility reasons).
        If all atoms have been found once (the returned list is of length len(atoms1)), this is
        set to True.

    Raises
    ------
    RuntimeError
        If some atoms have been found more than once, an exception is raised. This indicates an
        ill-defined Atoms object.

    """
    atomic_indices = []
    for a1 in atoms1:
        for a2 in atoms2:
            if (
                np.linalg.norm(a1.position - a2.position) < cutoff
                and a1.symbol == a2.symbol
            ):
                atomic_indices.append(a2.index)

    if len(atomic_indices) == len(atoms1):
        return atomic_indices, True
    elif len(atomic_indices) < len(atoms1):
        return atomic_indices, False
    else:
        raise RuntimeError(
            "More atoms found than looked for... Are there atoms unphysically close to each other, or duplicate atoms?"
        )


def copy_docstring(take_from_fct):
    """A decorator to copy the docstring of one function to a different function.

    Parameters
    ----------
    take_from_fct:
        The function, whose docstring shall be copied

    Returns
    -------
        The decorator

    Examples
    --------
    >>> def fun1():
    >>>     '''
    >>>     This function has a docstring!
    >>>     '''
    >>>     return 1
    >>>
    >>> @copy_docstring(fun1)
    >>> def fun2():
    >>>     return 2
    >>>
    >>> help(fun2)
    Help on function fun2 in module __main__:
    fun2()
    This function has a docstring!

    """
    docstring = take_from_fct.__doc__

    def decorator(give_to_fct):
        give_to_fct.__doc__ = docstring
        return give_to_fct

    return decorator


def redirect_stdout_to_file(filename):
    """
    Decorator to redirect standard output (stdout) to a file during the execution of a function.

    Parameters
    ----------
    filename : str
        The name of the file to which stdout will be redirected.

    Returns
    -------
    callable
        A decorator that takes a function and returns a wrapped function with stdout redirected.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            original_stdout = sys.stdout  # Save the original stdout
            with open(filename, "a") as f:  # Open the file
                sys.stdout = f  # Redirect stdout to the file
                result = func(*args, **kwargs)  # Call the decorated function
            sys.stdout = original_stdout  # Restore the original stdout
            return result

        return wrapper

    return decorator


######################################################################################################################
# The following functions are currently not used by RIIGID
# They are still stored here, for possible future purposes


def get_atoms_indices_by_height(all_atoms, middle_height, above=True, direction="z"):
    """The indices of atoms above/below middle_height are returned.

    A useful function to find the indices of an fragment or to set up a new fragment.

    above=True: Atoms above middle_height are considered and their indices in all_atoms are returned.

    above=False: Atoms below middle_height are considered and their indices in all_atoms are returned.

    Parameters
    ----------
    all_atoms: ase.atoms.Atoms
        The full Atoms object
    middle_height: number
        The height used to separate atoms; [Å]
    above: Bool, default: True
        See explanation given above
    direction: 'x','y' or 'z', default: 'z'
        The direction used to separate the atoms

    Returns
    -------
    list
        List containing indices of all atoms above/below middle_height

    Raises
    ------
    ValueError
        If the given input value for the 'direction' parameter is not known.

    """
    if direction == "x":
        direction = 0
    elif direction == "y":
        direction = 1
    elif direction == "z":
        direction = 2
    else:
        raise ValueError("Direction not known!")

    if above:
        return [
            atom.index
            for atom in all_atoms
            if atom.position[direction] >= middle_height
        ]
    else:
        return [
            atom.index for atom in all_atoms if atom.position[direction] < middle_height
        ]


######################################################################################################################
