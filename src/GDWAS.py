from copy import copy, deepcopy

from optimization_step import Optimization_Step
from optimizer import Optimizer


class GDWAS(Optimizer):
    """RIGID optimizer: Gradient Descent with Adaptive Stepsize

    Calculates force and torque on each fragment and moves them accordingly (like rigid bodies).

    The goal of this optimizer is to converge to the nearest minimum. To achieve this, in case the
    last update of the atomic positions lead to higher energies, it is dropped and the calculation
    is continued from the previous structure. This way, climbing upwards in the potential energy
    surface (PES) is prohibited and the probability for jumping into a different potential well is
    lowered.
    Additionally, a hard cap on the movement of individual atoms is enforced by max_step. This again
    helps in reducing the probability of the structure leaving a local minimum.

    The stepsize is adaptive and can change for two reasons:

    1.) If the last update performed on the atomic positions lowered the total energy (i.e a larger
    negative number), the stepsize is increased by multiplying with stepsize_factor_up.
    On the other hand, if the energy increased, the stepsize is lowered by multiplying with
    stepsize_factor_dn.

    2.) As described above, a hard cap on the movement of individual atoms is enforced by max_step.
    If an atom is found to move more than it is allowed to, the stepsize is lowered in such a way,
    that the maximal displacement of the atoms is exactly max_step.

    The user never needs to interact directly with the stepsize. Instead, via max_step_0, the user
    specifies how much the atoms (at most) shall move in the first optimization step. The stepsize
    is then initialized accordingly and from there on adapted automatically, as described above.

    Attributes
    ----------
    optimization_history: list of optimization_step.Optimization_Step
        The history of the optimization, which shall be checked for convergence.
    iteration: int
        Counts the number of finished optimization steps
    stepsize: number
        Timestep; [Da*Å**2/eV]
    stepsize_factor_up: number > 1
        Increase stepsize by this factor, if last optimization step lowered the total energy
    stepsize_factor_dn: number < 1
        Decrease stepsize by this factor, if last optimization step increased the total energy
    max_step: number
        The maximum distance atoms are allowed to move per optimization step; [Å]
    max_step_0: number
        In the first optimization step, the stepsize is chosen such that the atom(s) moving
        the farthest change their position by this value; [Å]
    start_structure : ase.atoms.Atoms
        The atoms forming the structure to be optimized.
        This is an ase.Atoms object and should include the
        correct unit cell (for periodic systems).
    calculator : ase.calculators.calculator.Calculator
        The used ASE calculator object
    convergence_criterion : convergence_criterion.Convergence_Criterion
        The used convergence criterion object
    current_structure: structure.Structure
        The structure currently used by the optimizer
    current_energy: number
        The energy of current_structure; [eV]
    current_forces: numpy.ndarray of shape (n_atoms_in_current_structure, 3)
        The forces in current_structure; [eV/Å]

    """

    def __init__(
        self,
        stepsize_factor_up=1.2,
        stepsize_factor_dn=0.2,
        max_step=0.1,
        max_step_0=0.01,
        *args,
        **kwargs,
    ):
        """Initialize the GDWAS optimizer.

        Parameters
        ----------
        stepsize_factor_up: number > 1, default: 1.2
            Increase stepsize by this factor, if last optimization step lowered the total energy
        stepsize_factor_dn: number < 1, default: 0.2
            Decrease stepsize by this factor, if last optimization step increased the total energy
        max_step: number, default: 0.1
            The maximum distance atoms are allowed to move per optimization step; [Å]
        max_step_0: number, default: 0.01
            In the first optimization step, the stepsize is chosen such that the atom(s) moving
            the farthest change their position by this value; [Å]

        """
        super().__init__()
        self.stepsize = 100  # initial stepsize, value doesn't matter
        self.stepsize_factor_up = stepsize_factor_up
        self.stepsize_factor_dn = stepsize_factor_dn
        self.max_step = max_step
        self.max_step_0 = max_step_0

    def run(self, start_structure, calculator, convergence_criterion, *args, **kwargs):
        """Let the optimizer run its optimization on the structure.

        Parameters
        ----------
        start_structure: structure.Structure
            The structure to be optimized
        calculator : ase.calculators.calculator.Calculator
            The used ASE calculator object
        convergence_criterion : convergence_criterion.Convergence_Criterion
            The used convergence criterion object

        """
        self.start_structure = start_structure
        self.calculator = calculator
        self.convergence_criterion = convergence_criterion

        while not convergence_criterion.is_converged:
            # Get current structure
            if self.iteration == 0:
                self.current_structure = deepcopy(start_structure)
            else:
                self.current_structure = deepcopy(
                    self.optimization_history[-1].updated_structure
                )

            # Do Calculation
            (
                self.current_energy,
                self.current_forces,
            ) = self.current_structure.calculate_energy_and_forces(
                calculator=calculator
            )

            # Adapt stepsize to energy change
            self.adapt_stepsize_to_energy_change()

            # undo last update if energy got larger
            self.drop_last_step_if_energy_got_larger()

            # Adapt stepsize again, if necessary (do test step to prevent too large movement)
            test_structure = deepcopy(self.current_structure)
            max_atomic_displacement_test, _ = test_structure.move(
                forces=self.current_forces, stepsize=self.stepsize
            )
            self.adapt_stepsize_to_prevent_too_large_steps(
                max_atomic_displacement=max_atomic_displacement_test
            )

            # Move atoms
            updated_structure = deepcopy(self.current_structure)
            _, _ = updated_structure.move(
                forces=self.current_forces, stepsize=self.stepsize
            )

            # Add Optimization step to history
            new_step = Optimization_Step(
                structure=self.current_structure,
                forces=self.current_forces,
                energy=self.current_energy,
                updated_structure=updated_structure,
            )
            self.optimization_history.append(new_step)

            # Prepare next iteration
            self.iteration += 1

            # Check for convergence
            self.convergence_criterion.check(
                optimization_history=self.optimization_history
            )

    def adapt_stepsize_to_energy_change(self):
        """Adapt the stepsize according to the last update step.

        If the last update performed on the atomic positions lowered the total energy (i.e a larger
        negative number), the stepsize is increased by multiplying with stepsize_factor_up.
        On the other hand, if the energy increased, the stepsize is lowered by multiplying with
        stepsize_factor_dn.

        """
        if self.iteration != 0:
            prev_energy = self.optimization_history[-1].energy
            current_energy = self.current_energy
            # Check if energy got smaller
            if current_energy < prev_energy:  # Yes -> make stepsize larger
                self.stepsize *= self.stepsize_factor_up
            else:  # No -> make stepsize smaller and drop latest optimization step
                self.stepsize *= self.stepsize_factor_dn

    def drop_last_step_if_energy_got_larger(self):
        """Drop last update step if it increased the total energy."""
        if self.iteration != 0:
            prev_energy = self.optimization_history[-1].energy
            current_energy = self.current_energy
            if prev_energy < current_energy:
                self.current_structure = deepcopy(
                    self.optimization_history[-1].structure
                )
                self.current_forces = copy(self.optimization_history[-1].forces)
                self.current_energy = copy(self.optimization_history[-1].energy)
                self.optimization_history.pop()

    def adapt_stepsize_to_prevent_too_large_steps(self, max_atomic_displacement):
        """Prevent too large atomic movement.

        If max_atomic_displacement > self.max_step, the stepsize is lowered, such that
        max_atomic_displacement = self.max_step.

        Parameters
        ----------
        max_atomic_displacement: number
            In the last update, how far did the atoms move at most (in rotations not all atoms
            move equally as far); [Å]

        """
        # Initialize stepsize or adapt it to too large atomic displacements
        if (
            self.iteration == 0
        ):  # In the first iteration we determine the initial stepsize, such that a step of max_step_0 is performed
            factor = self.max_step_0 / max_atomic_displacement
        else:  # In later iterations, calc factor to prevent too large atomic deplacements
            if max_atomic_displacement > self.max_step:
                factor = self.max_step / max_atomic_displacement
            else:
                factor = 1.0
        self.stepsize *= factor
