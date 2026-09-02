"""Test matrices for the SparseMatrixColorings.jl cross-validation suite.

The random size/density sweeps mirror ``test/random.jl`` in
SparseMatrixColorings.jl: small dense-ish matrices plus larger sparse ones,
in both tall and wide orientations.
The structured matrices add deterministic patterns whose coloring is easy to
reason about (diagonal, banded, arrow, block-diagonal) along with the two
worked examples from the coloring literature that SMC ships as fixtures.
"""

import numpy as np
from numpy.typing import NDArray

# (rows, columns, density) triples for non-symmetric colorings.
ASYMMETRIC_PARAMS = [
    *[(10, 20, p) for p in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)],
    *[(20, 10, p) for p in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)],
    *[(100, 200, p) for p in (0.01, 0.02, 0.03, 0.04, 0.05)],
    *[(200, 100, p) for p in (0.01, 0.02, 0.03, 0.04, 0.05)],
]

# (size, density) pairs for symmetric colorings.
SYMMETRIC_PARAMS = [
    *[(10, p) for p in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)],
    *[(100, p) for p in (0.01, 0.02, 0.03, 0.04, 0.05)],
]


def random_matrix(m: int, n: int, p: float) -> NDArray[np.int_]:
    """Random ``m x n`` 0/1 matrix with each entry non-zero with probability ``p``.

    The seed is derived from the shape and density,
    so a parametrized case always sees the same matrix.
    """
    rng = np.random.default_rng([m, n, round(p * 1000)])
    return (rng.random((m, n)) < p).astype(int)


def random_symmetric_matrix(n: int, p: float, *, diagonal: bool) -> NDArray[np.int_]:
    """Random structurally symmetric ``n x n`` 0/1 matrix.

    Symmetrized by ``max(M, M.T)``, so the off-diagonal density is slightly
    above ``p``.
    ``diagonal`` forces the diagonal fully on or fully off.
    """
    rng = np.random.default_rng([n, round(p * 1000), int(diagonal)])
    matrix = (rng.random((n, n)) < p).astype(int)
    matrix = np.maximum(matrix, matrix.T)
    np.fill_diagonal(matrix, int(diagonal))
    return matrix


def _banded(n: int, bandwidth: int) -> NDArray[np.int_]:
    """Square banded matrix with ``|i - j| <= bandwidth``."""
    offsets = np.subtract.outer(np.arange(n), np.arange(n))
    return (np.abs(offsets) <= bandwidth).astype(int)


def _arrow(n: int) -> NDArray[np.int_]:
    """Arrow matrix: full first row, full first column, full diagonal."""
    matrix = np.eye(n, dtype=int)
    matrix[0, :] = 1
    matrix[:, 0] = 1
    return matrix


def _block_diagonal(n: int, block: int) -> NDArray[np.int_]:
    """Block-diagonal matrix with dense ``block x block`` blocks."""
    matrix = np.zeros((n, n), dtype=int)
    for start in range(0, n, block):
        matrix[start : start + block, start : start + block] = 1
    return matrix


def _cycle_graph(n: int) -> NDArray[np.int_]:
    """Adjacency matrix of an ``n``-vertex cycle (no diagonal)."""
    matrix = _banded(n, 1) - np.eye(n, dtype=int)
    matrix[0, -1] = matrix[-1, 0] = 1
    return matrix


def _dense_row_and_col(m: int, n: int) -> NDArray[np.int_]:
    """Otherwise-empty matrix with one dense row and one dense column."""
    matrix = np.zeros((m, n), dtype=int)
    matrix[m // 2, :] = 1
    matrix[:, n // 2] = 1
    return matrix


def _what_fig_41() -> NDArray[np.int_]:
    """Symmetric pattern from Figure 4.1 of "What color is your Jacobian?".

    Same fixture as ``what_fig_41`` in SparseMatrixColorings.jl.
    """
    # fmt: off
    return np.array(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 1],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
        ]
    )
    # fmt: on


def _efficient_fig_1() -> NDArray[np.int_]:
    """Symmetric pattern from Figure 1 of "Efficient computation of sparse hessians".

    Same fixture as ``efficient_fig_1`` in SparseMatrixColorings.jl.
    """
    # fmt: off
    return np.array(
        [
            [1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
        ]
    )
    # fmt: on


# Deterministic rectangular patterns, exercised by row and column coloring.
STRUCTURED_MATRICES: dict[str, NDArray[np.int_]] = {
    "zeros_8x5": np.zeros((8, 5), dtype=int),
    "ones_6x9": np.ones((6, 9), dtype=int),
    "single_row_1x7": np.ones((1, 7), dtype=int),
    "single_col_7x1": np.ones((7, 1), dtype=int),
    "identity_10": np.eye(10, dtype=int),
    "antidiagonal_10": np.fliplr(np.eye(10, dtype=int)),
    "bidiagonal_12": np.eye(12, dtype=int) + np.eye(12, k=1, dtype=int),
    "tridiagonal_12": _banded(12, 1),
    "banded_15": _banded(15, 2),
    "arrow_10": _arrow(10),
    "block_diagonal_12": _block_diagonal(12, 3),
    "lower_triangular_8": np.tril(np.ones((8, 8), dtype=int)),
    "dense_row_and_col_9x7": _dense_row_and_col(9, 7),
    "what_fig_41": _what_fig_41(),
    "efficient_fig_1": _efficient_fig_1(),
}

# Deterministic structurally symmetric patterns, exercised by star coloring.
STRUCTURED_SYMMETRIC_MATRICES: dict[str, NDArray[np.int_]] = {
    name: matrix
    for name, matrix in STRUCTURED_MATRICES.items()
    if matrix.shape[0] == matrix.shape[1] and np.array_equal(matrix, matrix.T)
} | {
    "zeros_9": np.zeros((9, 9), dtype=int),
    "ones_7": np.ones((7, 7), dtype=int),
    "path_graph_8": _banded(8, 1) - np.eye(8, dtype=int),
    "cycle_graph_9": _cycle_graph(9),
}


def with_diagonal(matrix: NDArray[np.int_]) -> NDArray[np.int_]:
    """Copy of a square matrix with the diagonal filled in.

    Hessian patterns normally carry a diagonal,
    and a full diagonal is what makes star coloring and symmetric orthogonality
    equivalent, so the checker comparison relies on it.
    """
    filled = np.array(matrix, dtype=int, copy=True)
    np.fill_diagonal(filled, 1)
    return filled
