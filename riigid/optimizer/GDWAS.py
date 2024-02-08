from copy import copy, deepcopy
import sys

from riigid.optimization_step import OptimizationStep
from riigid.optimizer.optimizer import Optimizer


class GDWAS(Optimizer):
    """RIIGID optimizer: Gradient Descent with Adaptive Stepsize

    Calculates force and torque on each fragment and moves them accordingly (like rigid bodies).

    The goal of this optimizer is to converge to the nearest minimum. To achieve this, in case the
    last update of the atomic positions lead to higher energies, it is dropped and the calculation
    is continued from the previous structure. This way, climbing upwards in the potential energy
    surface (PES) is prohibited and the probability for jumping into a different potential well is
    lowered.
    Additionally, a hard cap on the movement of individual fragments is enforced by max_trans and max_rot. This again
    helps in reducing the probability of the structure leaving a local minimum.

    The stepsize is adaptive and can change for two reasons:

    1.) If the last update performed on the atomic positions lowered the total energy (i.e a larger
    negative number), the stepsize is increased by multiplying with stepsize_factor_up.
    On the other hand, if the energy increased, the stepsize is lowered by multiplying with
    stepsize_factor_dn.

    2.) As described above, a hard cap on the movement of individual fragments is enforced by max_trans and max_rot.
    If a fragment is found to translate more than max_trans, the stepsize is lowered in such a way,
    that the maximal displacement of the fragments is exactly max_trans.
    On the other hand, if a fragment is rotated by more than max_rot, the stepsize is adapted, such that
    the maximal rotation (i.e. the rotation of the fragment that rotates the most) is max_rot.

    The user never needs to interact directly with the stepsize. Instead, via max_trans_0 and max_rot_0, the user
    specifies how much the fragments (at most) shall move in the first optimization step. The stepsize
    is then initialized accordingly and from there on adapted automatically, as described above.

    Attributes
    ----------
    optimization_history: list of riigid.Optimization_Step
        The history of the optimization, which shall be checked for convergence.
    iteration: int
        Counts the number of finished optimization steps
    stepsize: number
        Timestep; [Da*Å**2/eV]
    stepsize_factor_up: number > 1
        Increase stepsize by this factor, if last optimization step lowered the total energy
    stepsize_factor_dn: number < 1
        Decrease stepsize by this factor, if last optimization step increased the total energy
    max_trans: number
        The maximum distance fragments are allowed to translate per optimization step; [Å]
    max_rot: number
        The maximum angle fragments are allowed to rotate per optimization step; [°]
    max_trans_0, max_rot_0: number, number
        In the first optimization step, the stepsize is chosen such that the fragment(s) translating/rotating
        the most, translate/rotate by (one of) these value; [Å], [°]
    start_with_random_step: bool
        Shall the fragments forming the structure be randomly translated and rotated before the
        first optimization step? This can be used to escape a saddle point starting-geometry.
        The attributes with '_r0' at the end further specify this random step before the first
        optimization step.
    displacement_r0: number
        How far shall the fragments be translated; [Å]
    angle_r0: number
        How much shall the fragments be rotated; [°]
    respect_restrictions_r0: bool
        If True, fragment.allowed_translation/rotation is respected.
        If False, rotation and translation in arbitrary directions is allowed temporarily.
        (After the random step, the restrictions are respected again.)
    seed_r0: int
        The random seed used to generate the translation directions and rotation axes
    max_iter: int
        The maximal number of optimization steps to be performed.
        If the calculation does not converge within this limit, it is stopped.
    start_structure: riigid.Structure
            The structure to be optimized
    calculator : ase.calculators.calculator.Calculator
        The used ASE calculator object
    convergence_criterion : riigid.convergence.Criterion
        The used convergence criterion object
    current_structure: riigid.Structure
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
        max_trans=0.1,
        max_rot=3,
        max_trans_0=0.01,
        max_rot_0=0.3,
        start_with_random_step=True,
        displacement_r0=0.001,
        angle_r0=0.03,
        respect_restrictions_r0=True,
        seed_r0=1234,
        max_iter=500,
    ):
        """Initialize the GDWAS optimizer.

        Parameters
        ----------
        stepsize_factor_up: number > 1, default: 1.2
            Increase stepsize by this factor, if last optimization step lowered the total energy
        stepsize_factor_dn: number < 1, default: 0.2
            Decrease stepsize by this factor, if last optimization step increased the total energy
        max_trans: number, default: 0.1
            The maximum distance fragments are allowed to translate per optimization step; [Å]
        max_rot: number, default: 3
            The maximum angle fragments are allowed to rotate per optimization step; [°]
        max_trans_0, max_rot_0: number, number, default: 0.01, 0.3
            In the first optimization step, the stepsize is chosen such that the fragment(s) translating/rotating
            the most, translate/rotate by (one of) these value; [Å], [°]
        start_with_random_step: bool, default:True
            Shall the fragments forming the structure be randomly translated and rotated before the
            first optimization step? This can be used to escape a saddle point starting-geometry.
            The parameters with '_r0' at the end further specify this random step before the first
            optimization step.
        displacement_r0: number, default:0.001
            How far shall the fragments be translated; [Å]
        angle_r0: number, default:0.03
            How much shall the fragments be rotated; [°]
        respect_restrictions_r0: bool, default:True
            If True, fragment.allowed_translation/rotation is respected.
            If False, rotation and translation in arbitrary directions is allowed temporarily.
            (After the random step, the restrictions are respected again.)
        seed_r0: int, default:1234
            The random seed used to generate the translation directions and rotation axes
        max_iter: int, default: 500
            The maximal number of optimization steps to be performed.
            If the calculation does not converge within this limit, it is stopped.

        """
        super().__init__(max_iter=max_iter)
        self.stepsize = 100  # initial stepsize, value doesn't matter, is later adapted to max_step_0
        self.stepsize_factor_up = stepsize_factor_up
        self.stepsize_factor_dn = stepsize_factor_dn
        self.max_trans = max_trans
        self.max_rot = max_rot
        self.max_trans_0 = max_trans_0
        self.max_rot_0 = max_rot_0
        self.start_with_random_step = start_with_random_step
        self.displacement_r0 = displacement_r0
        self.angle_r0 = angle_r0
        self.respect_restrictions_r0 = respect_restrictions_r0
        self.seed_r0 = seed_r0

    def run(self, start_structure, calculator, convergence_criterion, callback=None):
        """Let the optimizer run its optimization on the structure.

        Parameters
        ----------
        start_structure: riigid.Structure
            The structure to be optimized
        calculator : ase.calculators.calculator.Calculator
            The used ASE calculator object
        convergence_criterion : riigid.convergence.criterion
            The used convergence criterion object
        callback : function, default:None
            A callback function can be used to safe the optimization progress after each step.

        """
        print("Starting optimization...")
        sys.stdout.flush()  # Flush the output immediately
        self.start_structure = start_structure
        self.calculator = calculator
        self.convergence_criterion = convergence_criterion

        while not convergence_criterion.is_converged and self.iteration < self.max_iter:
            # Get current structure (starting structure or updated structure from last step)
            if self.iteration == 0:
                self.current_structure = deepcopy(start_structure)
            else:
                self.current_structure = deepcopy(
                    self.optimization_history[-1].updated_structure
                )

            # Before first calculation, perform a random step, if start_with_random_step==True
            if self.start_with_random_step and self.iteration == 0:
                self.current_structure.move_random_step(
                    displacement=self.displacement_r0,
                    angle=self.angle_r0,
                    respect_restrictions=self.respect_restrictions_r0,
                    seed=self.seed_r0,
                )

            # Do Calculation to get energy and forces
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

            if (
                self.iteration == 0
            ):  # Initialize stepsize accordingly to max_rot/trans_0
                self.initialize_stepsize_in_first_iteration()
            else:  # Adapt stepsize again, if necessary
                self.adapt_stepsize_to_prevent_too_large_steps()

            # Move atoms
            updated_structure = deepcopy(self.current_structure)
            _, _ = updated_structure.move(
                forces=self.current_forces, stepsize=self.stepsize
            )

            # Add Optimization step to history
            new_step = OptimizationStep(
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

            # If a callback function was provided, execute it. Useful to save data after every step
            if callback:
                callback()

        self.print_reason_for_end_of_optimization()

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
            else:  # No -> make stepsize smaller
                self.stepsize *= self.stepsize_factor_dn

    def drop_last_step_if_energy_got_larger(self):
        """Drop last update step if it increased the total energy."""
        if self.iteration != 0:
            prev_energy = self.optimization_history[-1].energy
            current_energy = self.current_energy
            if prev_energy < current_energy:
                # Return to structure, forces and energy, previous to the last update
                self.current_structure = deepcopy(
                    self.optimization_history[-1].structure
                )
                self.current_forces = copy(self.optimization_history[-1].forces)
                self.current_energy = copy(self.optimization_history[-1].energy)
                self.optimization_history.pop()

    def adapt_stepsize_to_prevent_too_large_steps(self):
        """Prevent too large movement of the fragments.

        If a fragment is found to translate more than self.max_trans, the stepsize is lowered in such a way,
        that the maximal displacement of the fragments is exactly self.max_trans.
        On the other hand, if a fragment is rotated by more than self.max_rot, the stepsize is adapted, such that
        the maximal rotation (i.e. the rotation of the fragment that rotates the most) is self.max_rot.
        (The weaker condition is fulfilled, i.e. the one with less movement/smaller stepsize.)

        Note
        ----
        Translation distances and rotation angles of fragments are directly proportional to the stepsize.
        This is used here to turn down too large stepsizes.

        """
        # Given the current stepsize, find the maximal rotation/translation of the fragments
        (
            max_found_translation_distance,
            max_found_angle,
        ) = self.current_structure.get_largest_translation_distance_and_largest_rotation_angle_from_forces(
            forces=self.current_forces, stepsize=self.stepsize
        )

        factor_rot = self.max_rot / max_found_angle
        factor_trans = self.max_trans / max_found_translation_distance
        factor = min([factor_rot, factor_trans])
        factor = min([1.0, factor])  # this function never makes the stepsize larger
        self.stepsize *= factor

    def initialize_stepsize_in_first_iteration(self):
        """Initialize the stepsize in the first iteration of the optimization.

        During the first iteration of the optimizations, this function adapts the stepsize
        such that the maximum displacement of the fragments is exactly self.max_trans_0 or the maximum
        rotation is exactly self.max_rot_0.
        (The weaker condition is fulfilled, i.e. the one with less movement/smaller stepsize.)

        Note
        ----
        Translation distances and rotation angles of fragments are directly proportional to the stepsize.
        This is used here to turn down too large stepsizes.

        """
        # Given the current stepsize, find the maximal rotation/translation of the fragments
        (
            max_found_translation_distance,
            max_found_angle,
        ) = self.current_structure.get_largest_translation_distance_and_largest_rotation_angle_from_forces(
            forces=self.current_forces, stepsize=self.stepsize
        )

        factor_rot = self.max_rot_0 / max_found_angle
        factor_trans = self.max_trans_0 / max_found_translation_distance
        factor = min([factor_rot, factor_trans])
        self.stepsize *= factor
