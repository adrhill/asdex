"""Check that asdex colorings match SparseMatrixColorings.jl exactly.

asdex's greedy colorings are ports of SparseMatrixColorings.jl (SMC).
Both sides use the ``LargestFirst`` vertex ordering and break ties by
ascending vertex index, so on the same pattern they must produce *identical*
color assignments, not merely colorings of the same quality.
Any divergence is a porting bug on one side, which is what these tests catch.

Julia numbers colors from 1, Python from 0, so SMC's colors are shifted down by
one before the comparison (see ``conftest.SMC``).
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from asdex import SparsityPattern
from asdex.coloring import color_cols, color_rows, color_symmetric
from tests.smc._matrices import (
    ASYMMETRIC_PARAMS,
    STRUCTURED_MATRICES,
    STRUCTURED_SYMMETRIC_MATRICES,
    SYMMETRIC_PARAMS,
    random_matrix,
    random_symmetric_matrix,
)

pytestmark = [pytest.mark.smc, pytest.mark.coloring]


def _assert_same_coloring(
    actual: NDArray[np.int32], num_colors: int, expected: NDArray[np.int32]
) -> None:
    """Assert asdex colors equal SMC colors, and that the color count agrees."""
    np.testing.assert_array_equal(actual, expected)
    expected_num = len(np.unique(expected[expected >= 0]))
    assert num_colors == expected_num


@pytest.mark.parametrize(("m", "n", "p"), ASYMMETRIC_PARAMS)
def test_color_cols_matches_smc_random(smc, m, n, p):
    """Column coloring of a random pattern matches SMC's column coloring."""
    matrix = random_matrix(m, n, p)
    colors, num_colors = color_cols(SparsityPattern.from_dense(matrix))
    _assert_same_coloring(colors, num_colors, smc.color_cols(matrix))


@pytest.mark.parametrize(("m", "n", "p"), ASYMMETRIC_PARAMS)
def test_color_rows_matches_smc_random(smc, m, n, p):
    """Row coloring of a random pattern matches SMC's row coloring."""
    matrix = random_matrix(m, n, p)
    colors, num_colors = color_rows(SparsityPattern.from_dense(matrix))
    _assert_same_coloring(colors, num_colors, smc.color_rows(matrix))


@pytest.mark.parametrize("postprocess", [False, True])
@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize(("n", "p"), SYMMETRIC_PARAMS)
def test_color_symmetric_matches_smc_random(smc, n, p, diagonal, postprocess):
    """Star coloring of a random symmetric pattern matches SMC's star coloring.

    Covers both postprocessing settings, since pruning unused colors is a
    separate pass whose neutral (``-1``) assignments must also agree.
    """
    matrix = random_symmetric_matrix(n, p, diagonal=diagonal)
    colors, num_colors, _ = color_symmetric(
        SparsityPattern.from_dense(matrix), postprocess=postprocess
    )
    expected = smc.color_symmetric(matrix, postprocess=postprocess)
    _assert_same_coloring(colors, num_colors, expected)


@pytest.mark.parametrize("name", list(STRUCTURED_MATRICES))
def test_color_cols_matches_smc_structured(smc, name):
    """Column coloring of a deterministic pattern matches SMC's column coloring."""
    matrix = STRUCTURED_MATRICES[name]
    colors, num_colors = color_cols(SparsityPattern.from_dense(matrix))
    _assert_same_coloring(colors, num_colors, smc.color_cols(matrix))


@pytest.mark.parametrize("name", list(STRUCTURED_MATRICES))
def test_color_rows_matches_smc_structured(smc, name):
    """Row coloring of a deterministic pattern matches SMC's row coloring."""
    matrix = STRUCTURED_MATRICES[name]
    colors, num_colors = color_rows(SparsityPattern.from_dense(matrix))
    _assert_same_coloring(colors, num_colors, smc.color_rows(matrix))


@pytest.mark.parametrize("postprocess", [False, True])
@pytest.mark.parametrize("name", list(STRUCTURED_SYMMETRIC_MATRICES))
def test_color_symmetric_matches_smc_structured(smc, name, postprocess):
    """Star coloring of a deterministic symmetric pattern matches SMC's."""
    matrix = STRUCTURED_SYMMETRIC_MATRICES[name]
    colors, num_colors, _ = color_symmetric(
        SparsityPattern.from_dense(matrix), postprocess=postprocess
    )
    expected = smc.color_symmetric(matrix, postprocess=postprocess)
    _assert_same_coloring(colors, num_colors, expected)
