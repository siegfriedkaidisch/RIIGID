from copy import copy, deepcopy
import sys
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from riigid.optimization_step import OptimizationStep
from riigid.optimizer.optimizer import Optimizer


class GPR(Optimizer):
    """RIIGID optimizer: Gaussian Process Regression

    blablabla

    Attributes
    ----------

    """

    def __init__(
        self,
    ):
        """Initialize the GPR optimizer.

        Parameters
        ----------

        """
        max_iter = 10
        super().__init__(max_iter=max_iter)
        # self.stepsize = 100

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

        x_range = (0, 10)
        num_steps = 6
        data_x = []
        data_y = []

        for i in range(num_steps):
            if i == 0:
                x = (x_range[1] - x_range[0]) / 2
            else:
                data_x_pred = np.linspace(x_range[0], x_range[1], 1000, endpoint=True)
                gaussian_process = GaussianProcessRegressor()
                gaussian_process.fit(data_x, data_y)
                mean_prediction, std_prediction = gaussian_process.predict(
                    data_x_pred, return_std=True
                )
                x = data_x_pred[np.argmax(std_prediction)]
            self.current_structure = deepcopy(start_structure)
            frag = self.current_structure.fragments[0]
            frag.translate_by_shift(shift=[x, 0, 0])

            # Do Calculation to get energy and forces
            (
                self.current_energy,
                self.current_forces,
            ) = self.current_structure.calculate_energy_and_forces(
                calculator=calculator
            )

            data_x.append(deepcopy(x))
            data_y.append(deepcopy(self.current_energy))
