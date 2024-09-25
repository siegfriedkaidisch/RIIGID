from copy import copy, deepcopy
import sys

from riigid.optimization_step import OptimizationStep
from riigid.optimizer.optimizer import Optimizer


class GD(Optimizer):
    """RIIGID optimizer: Gradient Descent

    Calculates force and torque on each fragment and moves them accordingly (like rigid bodies).
    The stepsize/timestep remains unchanged during the whole optimization. So, this is basically
    normal gradient descent.

    Attributes
    ----------
    optimization_history: list of riigid.Optimization_Step
        The history of the optimization, which shall be checked for convergence.
    iteration: int
        Counts the number of finished optimization steps
    stepsize: number
        Timestep; [Da*Å**2/eV]
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
        stepsize=1.0,
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
        stepsize: number, default:1.0
            Timestep; [Da*Å**2/eV]
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
        self.stepsize = stepsize
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
            print("Starting step " + str(self.iteration))
            sys.stdout.flush()

            # Get current structure (starting structure or updated structure from last step)
            if self.iteration == 0:
                self.current_structure = deepcopy(start_structure)
            else:
                self.current_structure = deepcopy(
                    self.optimization_history[-1].updated_structure
                )

            # Before first calculation, perform a random step, if start_with_random_step==True
            if self.start_with_random_step and self.iteration == 0:
                print("Doing random step before first calculation.")
                sys.stdout.flush()
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

            # Move atoms
            updated_structure = deepcopy(self.current_structure)
            _, _ = updated_structure.move(
                forces=self.current_forces, stepsize=self.stepsize
            )

            # Add Optimization step to history
            new_step = OptimizationStep(
                structure=self.current_structure,
                force_on_atoms=self.current_forces,
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
            if callback is not None:
                callback()

        self.print_reason_for_end_of_optimization()
