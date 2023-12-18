import pickle

from misc_functions import copy_docstring
from structure import Structure


class RIGID:
    """RIGID Geometry Optimization

    The structure under investigation is separated into so-called fragments, which are a
    set of atoms with fixed bonds between them, thus forming a rigid body. Using an ASE
    calculator (e.g. VASP), the forces on each atom are found, which are then used to
    calculate the force and torque on each fragment. Just like rigid bodies, the fragments
    are then moved in accordance to these forces and torques (like rigid bodies). This way,
    the energy of this system of (rigid) fragments is minimized.

    The user has to provide the structure, define the fragments and choose calculator,
    optimizer and convergence criterion.

    Attributes
    ----------
    start_structure : ase.atoms.Atoms
        The atoms forming the structure to be optimized.
        This is an ase.Atoms object and should include the
        correct unit cell (for periodic systems).
    calculator : ase.calculators.calculator.Calculator
        The used ASE calculator object
    optimizer : optimizer.Optimizer
        The used optimizer object
    convergence_criterion : convergence_criterion.Convergence_Criterion
        The used convergence criterion object

    """

    def __init__(self, atoms):
        """Initialize a RIGID geometry optimization.

        Parameters
        ----------
        atoms : ase.atoms.Atoms
            The atoms forming the structure to be optimized.
            This is an ase.Atoms object and should include the
            correct unit cell (for periodic systems).

        """
        self.start_structure = Structure(atoms=atoms)

    @copy_docstring(Structure.define_fragment_by_indices)
    def define_fragment_by_indices(self, *args, **kwargs):
        self.start_structure.define_fragment_by_indices(*args, **kwargs)

    def set_calculator(self, calculator):
        """Set the ASE Calculator to be used for optimizing the structure.

        Parameters
        ----------
        calculator : ase.calculators.calculator.Calculator
            The used ASE calculator object

        """
        self.calculator = calculator

    def set_optimizer(self, optimizer):
        """Set the optimizer to be used for optimizing the structure.

        Parameters
        ----------
        optimizer : optimizer.Optimizer
            The used optimizer object

        """
        self.optimizer = optimizer

    def set_convergence_criterion(self, convergence_criterion):
        """Set the convergence criterion for optimizing the structure.

        Parameters
        ----------
        convergence_criterion : convergence_criterion.Convergence_Criterion
            The used convergence criterion object

        """
        self.convergence_criterion = convergence_criterion

    def run(self):
        """Run the optimization"""
        # Perform rigid optimization
        self.optimizer.run(
            start_structure=self.structure,
            calculator=self.calculator,
            convergence_criterion=self.convergence_criterion,
        )

        # Save and print results
        self.save_optimization_data()  # get raw data (optimization history) from optimizer
        self.print_optimization_summary()  # get raw data from optimizer

    def save_optimization_data(self, filename):
        """
        ??
        """
        optimization_history = self.optimizer.optimization_history
        f = open(filename, "wb")
        pickle.dump(optimization_history, f)
        f.close()

    def print_optimization_summary(self, properties):
        """
        ??
        properties defines which properties will be printed
        extra function regarding final step
        extra function regarding logging this data to a txt file
        """
        optimization_history = self.optimizer.optimization_history
        energies = [step.energy for step in optimization_history]
        print("Energies: ")  # unit?
        print(energies)
