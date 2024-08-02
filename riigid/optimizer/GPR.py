from copy import copy, deepcopy
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel, ConstantKernel

from riigid.optimization_step import OptimizationStep
from riigid.optimizer.optimizer import Optimizer


class GPR(Optimizer):
    """RIIGID optimizer: Gaussian Process Regression

    NOT yet ready for use.

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
        num_steps = 20
        data_x = []
        data_y = []
        data_x_pred = np.linspace(x_range[0], x_range[1], 1000, endpoint=True)

        for i in range(num_steps):
            if i == 0:
                x = (x_range[1] - x_range[0]) / 2
            else:
                kernel = RBF()
                gaussian_process = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
                gaussian_process.fit(np.array(data_x).reshape(-1, 1), np.array(data_y).reshape(-1))
                mean_prediction, std_prediction = gaussian_process.predict(
                    data_x_pred.reshape(-1, 1), return_std=True
                )
                x = data_x_pred[np.argmax(std_prediction)]
            self.current_structure = deepcopy(start_structure)
            self.current_structure.shift_and_rotate_a_fragment(fragment_index=0, shift=[x, 0, 0], angle=0, axis=[1,1,1])

            # Do Calculation to get energy and forces
            (
                self.current_energy,
                self.current_forces,
            ) = self.current_structure.calculate_energy_and_forces(
                calculator=deepcopy(calculator)
            )

            data_x.append(deepcopy(x))
            data_y.append(deepcopy(self.current_energy))

            # Plots
            kernel = RBF()
            gaussian_process = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
            gaussian_process.fit(np.array(data_x).reshape(-1, 1), np.array(data_y).reshape(-1))
            #print(gaussian_process.kernel_)
            #print(data_x)
            #print(data_y)
            mean_prediction, std_prediction = gaussian_process.predict(
                data_x_pred.reshape(-1, 1), return_std=True
            )
            plt.figure()
            plt.scatter(data_x, np.array(data_y), label="Calculations")
            plt.plot(data_x_pred, mean_prediction, label="Mean prediction")
            plt.fill_between(
                data_x_pred,
                mean_prediction - 1.96 * std_prediction,
                mean_prediction + 1.96 * std_prediction,
                alpha=0.5,
                label=r"95% confidence interval",
            )
            plt.legend()
            plt.xlabel("$x [A]$")
            plt.ylabel("$Energy [eV]$")
            plt.ylim(np.min(mean_prediction - 1.96 * std_prediction), np.max(mean_prediction + 1.96 * std_prediction))
            plt.title("Gaussian process regression")
            plt.savefig("./gpr_pred" + str(i) + ".svg")
