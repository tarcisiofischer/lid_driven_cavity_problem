import os

import pytest

from lid_driven_cavity_problem.nonlinear_solver._common import _calculate_jacobian_mask
import numpy as np


PLOT = False
GENERATE = False

@pytest.mark.parametrize(
    ("nx, ny, dof"),
    (
        (5, 5, 1),
        (4, 4, 3),
        (4, 2, 3),
        (2, 4, 3),
    )
)
def test_calculate_jacobian_mask(nx, ny, dof):
    expected_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_calculate_jacobian_mask')

    mask = _calculate_jacobian_mask(nx, ny, dof)
    expected_mask_filename = os.path.join(expected_path, 'J_%s_%s_%s.txt' % (nx, ny, dof))

    if GENERATE:
        np.savetxt(expected_mask_filename, mask)
        assert False, "Generation finished. Failing test (This is expected behavior)"

    expected_mask = np.loadtxt(expected_mask_filename)

    if PLOT:
        import matplotlib.pyplot as plt
        plt.imshow(mask, aspect='equal', interpolation='none')
        ax = plt.gca()
        ax.set_xticks(np.arange(-.5, len(mask) - .5, 1))
        ax.set_yticks(np.arange(-.5, len(mask) - .5, 1))
        ax.grid(color='black', linestyle='-', linewidth=2)

        plt.show()
        assert False, "Plotting finished. Failing test (This is expected behavior)"

    assert np.allclose(mask, expected_mask)
