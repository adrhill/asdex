"""Bridge from pytest to SparseMatrixColorings.jl via PythonCall.jl.

The Julia runtime is started lazily inside the session-scoped ``smc`` fixture,
never at import time, so collecting the core test suite (which deselects the
``smc`` marker) never loads Julia.

Julia numbers colors from 1 and neutral vertices with 0,
while asdex numbers colors from 0 and neutral vertices with -1;
every color vector crossing the bridge is shifted accordingly.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

# Package registered in the juliacall environment before Julia starts.
_SMC_NAME = "SparseMatrixColorings"
_SMC_UUID = "0a514795-09f3-496d-8182-132a7b665d35"
_SMC_VERSION = "0.4"

# Julia-side helpers.
# Matrices cross the bridge as 0-based COO index arrays plus a shape,
# which sidesteps the row-major/column-major mismatch of dense arrays.
_JULIA_SETUP = """
module AsdexSMC

using SparseArrays
using SparseMatrixColorings
using SparseMatrixColorings:
    structurally_orthogonal_columns, symmetrically_orthogonal_columns

matrix(rows, cols, m, n) = sparse(
    Int.(rows) .+ 1, Int.(cols) .+ 1, ones(Float64, length(rows)), Int(m), Int(n)
)

algorithm(postprocessing) = GreedyColoringAlgorithm(
    LargestFirst(); postprocessing=Bool(postprocessing), decompression=:direct
)

function color_cols(rows, cols, m, n)
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:column)
    result = coloring(matrix(rows, cols, m, n), problem, algorithm(false))
    return collect(column_colors(result))
end

function color_rows(rows, cols, m, n)
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:row)
    result = coloring(matrix(rows, cols, m, n), problem, algorithm(false))
    return collect(row_colors(result))
end

function color_symmetric(rows, cols, n, postprocessing)
    A = matrix(rows, cols, n, n)
    @assert A == transpose(A)
    problem = ColoringProblem(; structure=:symmetric, partition=:column)
    result = coloring(A, problem, algorithm(postprocessing))
    return collect(column_colors(result))
end

function orthogonal_cols(rows, cols, m, n, colors)
    return structurally_orthogonal_columns(matrix(rows, cols, m, n), Int.(colors) .+ 1)
end

function orthogonal_symmetric(rows, cols, n, colors)
    A = Matrix(matrix(rows, cols, n, n))
    return symmetrically_orthogonal_columns(A, Int.(colors) .+ 1)
end

end
"""


class SMC:
    """Wrapper around the SparseMatrixColorings.jl entry points we compare against.

    Every method takes a dense 0/1 matrix and returns 0-based colors, so that
    results line up with asdex without any further index juggling at the call site.
    """

    def __init__(self, module):
        """Wrap the ``AsdexSMC`` Julia module holding the helper functions."""
        self._jl = module

    def color_cols(self, dense: NDArray) -> NDArray[np.int32]:
        """SMC column coloring with ``LargestFirst``, as 0-based colors."""
        rows, cols = _coo(dense)
        m, n = dense.shape
        return _shift(self._jl.color_cols(rows, cols, m, n))

    def color_rows(self, dense: NDArray) -> NDArray[np.int32]:
        """SMC row coloring with ``LargestFirst``, as 0-based colors."""
        rows, cols = _coo(dense)
        m, n = dense.shape
        return _shift(self._jl.color_rows(rows, cols, m, n))

    def color_symmetric(
        self, dense: NDArray, *, postprocess: bool
    ) -> NDArray[np.int32]:
        """SMC star coloring with ``LargestFirst``, as 0-based colors.

        Pruned (neutral) vertices come back as ``-1``, matching asdex.
        """
        rows, cols = _coo(dense)
        n = dense.shape[0]
        return _shift(self._jl.color_symmetric(rows, cols, n, postprocess))

    def orthogonal_cols(self, dense: NDArray, colors: NDArray) -> bool:
        """Whether the 0-based column coloring is structurally orthogonal."""
        rows, cols = _coo(dense)
        m, n = dense.shape
        return bool(self._jl.orthogonal_cols(rows, cols, m, n, _as_int64(colors)))

    def orthogonal_rows(self, dense: NDArray, colors: NDArray) -> bool:
        """Whether the 0-based row coloring is structurally orthogonal.

        SMC only exposes the column-wise check, so the matrix is transposed.
        """
        return self.orthogonal_cols(np.asarray(dense).T, colors)

    def orthogonal_symmetric(self, dense: NDArray, colors: NDArray) -> bool:
        """Whether the 0-based coloring is symmetrically orthogonal."""
        rows, cols = _coo(dense)
        n = dense.shape[0]
        return bool(self._jl.orthogonal_symmetric(rows, cols, n, _as_int64(colors)))


def _coo(dense: NDArray) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Split a dense 0/1 matrix into 0-based row and column index arrays."""
    rows, cols = np.nonzero(np.asarray(dense))
    return rows.astype(np.int64), cols.astype(np.int64)


def _as_int64(colors: NDArray) -> NDArray[np.int64]:
    """Coerce a color vector to the integer type the Julia helpers expect."""
    return np.asarray(colors, dtype=np.int64)


def _shift(colors) -> NDArray[np.int32]:
    """Convert Julia's 1-based colors (0 = neutral) to asdex's 0-based (-1 = neutral)."""
    return np.asarray(colors, dtype=np.int32) - 1


@pytest.fixture(scope="session")
def smc() -> SMC:
    """Start Julia, install SparseMatrixColorings.jl, and expose its coloring API."""
    juliapkg = pytest.importorskip(
        "juliapkg", reason="install the 'smc' dependency group to run these tests"
    )
    juliapkg.require_julia("1.10")
    juliapkg.add(_SMC_NAME, _SMC_UUID, version=_SMC_VERSION)
    juliapkg.resolve()

    # Imported here rather than at module scope: importing juliacall boots a
    # Julia runtime, and only tests using this fixture should pay that cost.
    from juliacall import Main as jl  # noqa: PLC0415

    jl.seval(_JULIA_SETUP)
    return SMC(jl.AsdexSMC)
