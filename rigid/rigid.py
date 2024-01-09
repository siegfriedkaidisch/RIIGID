import pickle
from ase.io.trajectory import Trajectory

from rigid.library.misc import copy_docstring
from rigid.structure import Structure


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
    name : str
        The name of the studied system.
    calculator : ase.calculators.calculator.Calculator
        The used ASE calculator object
    optimizer : optimizer.Optimizer
        The used optimizer object
    convergence_criterion : convergence_criterion.Convergence_Criterion
        The used convergence criterion object

    """

    def __init__(self, atoms, name):
        """Initialize a RIGID geometry optimization.

        Parameters
        ----------
        atoms : ase.atoms.Atoms
            The atoms forming the structure to be optimized.
            This is an ase.Atoms object and should include the
            correct unit cell (for periodic systems).
        name : str
            The name of the studied system. E.g.: "Benzene"

        """
        self.start_structure = Structure(atoms=atoms)
        self.name = name

    @copy_docstring(Structure.define_fragment_by_indices)
    def define_fragment_by_indices(self, *args, **kwargs):
        self.start_structure.define_fragment_by_indices(*args, **kwargs)

    def set_calculator(self, calculator, settings=None):
        """Set the ASE Calculator to be used for optimizing the structure.

        Parameters
        ----------
        calculator : ase.calculators.calculator.Calculator
            The used ASE calculator object

        """
        # if calculator.lower() == "vasp"
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
            start_structure=self.start_structure,
            calculator=self.calculator,
            convergence_criterion=self.convergence_criterion,
        )

        # Save some results
        self.save_optimization_history() 
        self.create_trajectory_file_from_optimization_history()

        # Print some results
        self.print_optimization_summary() 

    def save_optimization_history(self):
        """Save the optimization history (list of optimization steps) as a pickle file."""
        optimization_history = self.optimizer.optimization_history
        f = open(self.name+'.pk', "wb")
        pickle.dump(optimization_history, f)
        f.close()

    def create_trajectory_file_from_optimization_history(self):
        """Creates and saves the trajectory file of the optimization."""
        optimization_history = self.optimizer.optimization_history
        traj = Trajectory(self.name+'.traj', 'w')
        for optimization_step in optimization_history:
            traj.write(optimization_step.structure.atoms)
        traj.close()

    def print_optimization_summary(self):
        """Print Information about the Optimization."""
        optimization_history = self.optimizer.optimization_history
        energies = [step.energy for step in optimization_history]
        print("Energies [eV]: ")
        print(energies)

    

