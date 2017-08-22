from scipy.optimize.slsqp import approx_jacobian

from lid_driven_cavity_problem.residual_function import residual_function
import numpy as np


def _plot_jacobian(graph, X, plot_lines_on_variables=False, plot_lines_on_elements=False):
    import matplotlib.pyplot as plt

    J = approx_jacobian(X, residual_function, 1e-4, graph)
    J = J.astype(dtype='bool')

    plt.imshow(J, aspect='equal', interpolation='none')
    ax = plt.gca()

    if plot_lines_on_variables:
        ax.set_xticks(np.arange(-.5, len(J), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(J), 1), minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

    if plot_lines_on_elements:
        ax.set_xticks(np.arange(-.5, len(J), 3), minor=False)
        ax.set_yticks(np.arange(-.5, len(J), 3), minor=False)
        ax.grid(color='r', linestyle='-', linewidth=2)

    plt.show()
