"""Tests for graph coloring algorithms.

Test cases inspired by SparseMatrixColorings.jl (MIT license)
Copyright (c) 2024 Guillaume Dalle, Alexis Montoison, and contributors
https://github.com/gdalle/SparseMatrixColorings.jl
See also: Dalle & Montoison (2025), https://arxiv.org/abs/2505.07308
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO
from numpy.testing import assert_allclose

from asdex import (
    ColoredPattern,
    DenseColoringWarning,
    SparsityPattern,
    check_coloring_cols,
    check_coloring_rows,
    check_coloring_symmetric,
    hessian_coloring,
    hessian_coloring_from_sparsity,
    hessian_from_coloring,
    jacobian_coloring,
    jacobian_coloring_from_sparsity,
)
from asdex._display import _compressed_pattern
from asdex.coloring import (
    InvalidColoringError,
    StarSet,
    color_cols,
    color_rows,
    color_symmetric,
)


def _make_banded(n: int, half_bandwidth: int) -> SparsityPattern:
    """Symmetric banded matrix with given half-bandwidth.

    Matches SparseMatrixColorings.jl's ``banded_matrix(n, 2*half_bandwidth)``.
    """
    rows, cols = [], []
    for i in range(n):
        for k in range(-half_bandwidth, half_bandwidth + 1):
            j = i + k
            if 0 <= j < n:
                rows.append(i)
                cols.append(j)
    return SparsityPattern.from_coo(rows, cols, (n, n))


def _make_symmetric_reflexive_graph(
    n: int, edges: list[tuple[int, int]]
) -> SparsityPattern:
    """Adjacency pattern of a reflexive undirected graph.

    Includes a self-loop at every vertex (diagonal) and both directions of
    each undirected edge.
    """
    rows = list(range(n)) + [i for i, j in edges] + [j for i, j in edges]
    cols = list(range(n)) + [j for i, j in edges] + [i for i, j in edges]
    return SparsityPattern.from_coo(rows, cols, (n, n))


def _make_arrow(n: int) -> SparsityPattern:
    """Arrow matrix: diagonal + dense first row/column."""
    rows, cols = [], []
    for i in range(n):
        rows.append(i)
        cols.append(i)  # diagonal
        if i > 0:
            rows.append(0)
            cols.append(i)  # first row
            rows.append(i)
            cols.append(0)  # first col
    return SparsityPattern.from_coo(rows, cols, (n, n))


# Row coloring tests


@pytest.mark.coloring
def test_diagonal_one_color():
    """Diagonal matrix: all rows are independent, should use 1 color."""
    sparsity = SparsityPattern.from_coo([0, 1, 2, 3], [0, 1, 2, 3], (4, 4))

    colors, num_colors = color_rows(sparsity)

    assert num_colors == 1
    assert len(colors) == 4
    assert np.all(colors == 0)
    check_coloring_rows(sparsity, colors)


@pytest.mark.coloring
def test_dense_m_colors():
    """Dense matrix: every row conflicts with every other, needs m colors."""
    rows, cols = [], []
    for i in range(4):
        for j in range(4):
            rows.append(i)
            cols.append(j)
    sparsity = SparsityPattern.from_coo(rows, cols, (4, 4))

    colors, num_colors = color_rows(sparsity)

    assert num_colors == 4
    assert len(colors) == 4
    assert len(set(colors)) == 4  # All different colors
    check_coloring_rows(sparsity, colors)


@pytest.mark.coloring
def test_block_diagonal():
    """Block diagonal: non-overlapping blocks can share colors."""
    # Two 2x2 blocks
    rows = [0, 0, 1, 1, 2, 2, 3, 3]
    cols = [0, 1, 0, 1, 2, 3, 2, 3]
    sparsity = SparsityPattern.from_coo(rows, cols, (4, 4))

    colors, num_colors = color_rows(sparsity)

    assert num_colors == 2
    check_coloring_rows(sparsity, colors)
    # Rows 0,1 conflict; rows 2,3 conflict; but 0,2 and 1,3 don't
    assert colors[0] != colors[1]
    assert colors[2] != colors[3]


@pytest.mark.coloring
def test_tridiagonal():
    """Tridiagonal matrix: needs 2-3 colors depending on structure."""
    # 4x4 tridiagonal
    rows = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
    cols = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3]
    sparsity = SparsityPattern.from_coo(rows, cols, (4, 4))

    colors, num_colors = color_rows(sparsity)

    # Tridiagonal needs at most 3 colors (greedy may use 2-3)
    assert 2 <= num_colors <= 3
    check_coloring_rows(sparsity, colors)


@pytest.mark.coloring
def test_single_row():
    """Single row matrix."""
    sparsity = SparsityPattern.from_coo([0, 0, 0], [0, 1, 2], (1, 3))

    colors, num_colors = color_rows(sparsity)

    assert num_colors == 1
    assert len(colors) == 1
    assert colors[0] == 0


@pytest.mark.coloring
def test_single_column():
    """Single column matrix: all rows conflict."""
    sparsity = SparsityPattern.from_coo([0, 1, 2], [0, 0, 0], (3, 1))

    colors, num_colors = color_rows(sparsity)

    assert num_colors == 3
    assert len(set(colors)) == 3
    check_coloring_rows(sparsity, colors)


@pytest.mark.coloring
def test_empty_matrix():
    """Empty matrix (0 rows)."""
    sparsity = SparsityPattern.from_coo([], [], (0, 3))

    colors, num_colors = color_rows(sparsity)

    assert num_colors == 0
    assert len(colors) == 0


@pytest.mark.coloring
def test_zero_matrix():
    """Matrix with no non-zeros: all rows independent."""
    sparsity = SparsityPattern.from_coo([], [], (3, 3))

    colors, num_colors = color_rows(sparsity)

    assert num_colors == 1
    assert len(colors) == 3
    assert np.all(colors == 0)


@pytest.mark.coloring
def test_lower_triangular():
    """Lower triangular: increasing conflicts per row."""
    # 4x4 lower triangular
    rows = []
    cols = []
    for i in range(4):
        for j in range(i + 1):
            rows.append(i)
            cols.append(j)
    sparsity = SparsityPattern.from_coo(rows, cols, (4, 4))

    colors, num_colors = color_rows(sparsity)

    check_coloring_rows(sparsity, colors)
    # Lower triangular needs 4 colors (row 3 conflicts with all)
    assert num_colors == 4


@pytest.mark.coloring
def test_checkerboard():
    """Checkerboard pattern: alternating rows/cols."""
    # 4x4 checkerboard (even rows: even cols, odd rows: odd cols)
    rows = []
    cols = []
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                rows.append(i)
                cols.append(j)
    sparsity = SparsityPattern.from_coo(rows, cols, (4, 4))

    colors, num_colors = color_rows(sparsity)

    check_coloring_rows(sparsity, colors)
    # Even rows share cols 0,2; odd rows share cols 1,3
    # So we need 2 colors
    assert num_colors == 2


@pytest.mark.coloring
def test_largest_first_improves_coloring():
    """LargestFirst achieves optimal coloring on bridged cliques.

    Two 3-cliques (rows {0,1,2} via col 0, rows {3,4,5} via col 1)
    bridged by col 2 (rows 0 and 3).
    Chromatic number is 3.
    LargestFirst colors the high-degree bridge vertices (0, 3) first,
    allowing the cliques to share colors optimally.
    """
    rows = [0, 1, 2, 3, 4, 5, 0, 3]
    cols = [0, 0, 0, 1, 1, 1, 2, 2]
    sparsity = SparsityPattern.from_coo(rows, cols, (6, 3))

    colors, num_colors = color_rows(sparsity)

    check_coloring_rows(sparsity, colors)
    assert num_colors == 3


@pytest.mark.coloring
def test_row_anti_diagonal():
    """Anti-diagonal: all rows are independent, 1 color suffices.

    From SMC small.jl.
    """
    sparsity = SparsityPattern.from_dense(
        np.array(
            [
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
            ]
        )
    )

    colors, num_colors = color_rows(sparsity)

    assert num_colors == 1
    check_coloring_rows(sparsity, colors)


@pytest.mark.coloring
def test_row_triangle():
    """Triangle pattern: complete bipartite-like, needs 3 colors.

    From SMC small.jl.
    """
    sparsity = SparsityPattern.from_dense(
        np.array(
            [
                [1, 1, 0],
                [0, 1, 1],
                [1, 0, 1],
            ]
        )
    )

    colors, num_colors = color_rows(sparsity)

    assert num_colors == 3
    check_coloring_rows(sparsity, colors)


@pytest.mark.coloring
def test_row_smc_small():
    """SMC small.jl row coloring test matrix: [1 0 1; 0 1 0; 1 1 0].

    SMC gets 2 colors with LargestFirst.
    """
    sparsity = SparsityPattern.from_dense(
        np.array(
            [
                [1, 0, 1],
                [0, 1, 0],
                [1, 1, 0],
            ]
        )
    )

    colors, num_colors = color_rows(sparsity)

    assert num_colors == 2
    check_coloring_rows(sparsity, colors)


@pytest.mark.coloring
def test_row_bidiagonal():
    """Upper bidiagonal 6x6: needs 2 colors.

    From SMC structured.jl.
    """
    sparsity = SparsityPattern.from_dense(
        np.array(
            [
                [1, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 1],
            ]
        )
    )

    colors, num_colors = color_rows(sparsity)

    assert num_colors == 2
    check_coloring_rows(sparsity, colors)


# Column coloring tests


@pytest.mark.coloring
def test_col_diagonal_one_color():
    """Diagonal matrix: all columns are independent, should use 1 color."""
    sparsity = SparsityPattern.from_coo([0, 1, 2, 3], [0, 1, 2, 3], (4, 4))

    colors, num_colors = color_cols(sparsity)

    assert num_colors == 1
    assert len(colors) == 4
    assert np.all(colors == 0)
    check_coloring_cols(sparsity, colors)


@pytest.mark.coloring
def test_col_dense_n_colors():
    """Dense matrix: every column conflicts with every other, needs n colors."""
    rows, cols = [], []
    for i in range(4):
        for j in range(4):
            rows.append(i)
            cols.append(j)
    sparsity = SparsityPattern.from_coo(rows, cols, (4, 4))

    colors, num_colors = color_cols(sparsity)

    assert num_colors == 4
    assert len(set(colors)) == 4
    check_coloring_cols(sparsity, colors)


@pytest.mark.coloring
def test_col_single_row():
    """Single row: all columns conflict."""
    sparsity = SparsityPattern.from_coo([0, 0, 0], [0, 1, 2], (1, 3))

    colors, num_colors = color_cols(sparsity)

    assert num_colors == 3
    assert len(set(colors)) == 3
    check_coloring_cols(sparsity, colors)


@pytest.mark.coloring
def test_col_single_column():
    """Single column: only one column, needs 1 color."""
    sparsity = SparsityPattern.from_coo([0, 1, 2], [0, 0, 0], (3, 1))

    colors, num_colors = color_cols(sparsity)

    assert num_colors == 1
    assert len(colors) == 1
    assert colors[0] == 0


@pytest.mark.coloring
def test_col_block_diagonal():
    """Block diagonal: non-overlapping blocks can share colors."""
    rows = [0, 0, 1, 1, 2, 2, 3, 3]
    cols = [0, 1, 0, 1, 2, 3, 2, 3]
    sparsity = SparsityPattern.from_coo(rows, cols, (4, 4))

    colors, num_colors = color_cols(sparsity)

    assert num_colors == 2
    check_coloring_cols(sparsity, colors)


@pytest.mark.coloring
def test_col_empty():
    """Empty columns."""
    sparsity = SparsityPattern.from_coo([], [], (3, 0))

    colors, num_colors = color_cols(sparsity)

    assert num_colors == 0
    assert len(colors) == 0


@pytest.mark.coloring
def test_col_tridiagonal():
    """Tridiagonal: column coloring also needs 2-3 colors."""
    rows = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
    cols = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3]
    sparsity = SparsityPattern.from_coo(rows, cols, (4, 4))

    colors, num_colors = color_cols(sparsity)

    assert 2 <= num_colors <= 3
    check_coloring_cols(sparsity, colors)


@pytest.mark.coloring
def test_col_anti_diagonal():
    """Anti-diagonal: all columns are independent, 1 color suffices.

    From SMC small.jl.
    """
    sparsity = SparsityPattern.from_dense(
        np.array(
            [
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
            ]
        )
    )

    colors, num_colors = color_cols(sparsity)

    assert num_colors == 1
    check_coloring_cols(sparsity, colors)


@pytest.mark.coloring
def test_col_triangle():
    """Triangle pattern: needs 3 column colors.

    From SMC small.jl.
    """
    sparsity = SparsityPattern.from_dense(
        np.array(
            [
                [1, 1, 0],
                [0, 1, 1],
                [1, 0, 1],
            ]
        )
    )

    colors, num_colors = color_cols(sparsity)

    assert num_colors == 3
    check_coloring_cols(sparsity, colors)


@pytest.mark.coloring
def test_col_smc_small():
    """SMC small.jl column coloring test matrix: [1 0 1; 0 1 1; 1 0 0].

    SMC gets 2 colors with LargestFirst.
    """
    sparsity = SparsityPattern.from_dense(
        np.array(
            [
                [1, 0, 1],
                [0, 1, 1],
                [1, 0, 0],
            ]
        )
    )

    colors, num_colors = color_cols(sparsity)

    assert num_colors == 2
    check_coloring_cols(sparsity, colors)


@pytest.mark.coloring
def test_col_bidiagonal():
    """Upper bidiagonal 6x6: needs 2 column colors.

    From SMC structured.jl.
    """
    sparsity = SparsityPattern.from_dense(
        np.array(
            [
                [1, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 1],
            ]
        )
    )

    colors, num_colors = color_cols(sparsity)

    assert num_colors == 2
    check_coloring_cols(sparsity, colors)


# Star coloring tests


@pytest.mark.coloring
def test_star_diagonal():
    """Diagonal Hessian: no off-diagonal entries, 1 color suffices."""
    sparsity = SparsityPattern.from_coo([0, 1, 2, 3], [0, 1, 2, 3], (4, 4))

    colors, num_colors, _ = color_symmetric(sparsity)

    assert num_colors == 1
    check_coloring_symmetric(sparsity, colors)


@pytest.mark.coloring
def test_star_dense():
    """Dense symmetric pattern: star coloring is valid."""
    rows, cols = [], []
    for i in range(4):
        for j in range(4):
            rows.append(i)
            cols.append(j)
    sparsity = SparsityPattern.from_coo(rows, cols, (4, 4))

    colors, num_colors, _ = color_symmetric(sparsity)

    check_coloring_symmetric(sparsity, colors)
    # Dense 4x4 needs at least 4 colors for distance-1
    assert num_colors >= 4


@pytest.mark.coloring
def test_star_tridiagonal():
    """Tridiagonal Hessian: star chromatic number is 3.

    Verified against SMC with LargestFirst.
    """
    rows = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
    cols = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3]
    sparsity = SparsityPattern.from_coo(rows, cols, (4, 4))

    colors, num_colors, _ = color_symmetric(sparsity)

    check_coloring_symmetric(sparsity, colors)
    assert num_colors == 3


@pytest.mark.coloring
def test_star_arrow_matrix():
    """Arrow matrix: star coloring needs only 2 colors.

    Row coloring needs n colors (all rows conflict via col 0),
    but the star graph has star chromatic number 2.
    Verified against SMC: star=2, row=10 for n=10.
    """
    sparsity = _make_arrow(10)

    star_colors, star_num, _ = color_symmetric(sparsity)
    row_colors, row_num = color_rows(sparsity)

    check_coloring_symmetric(sparsity, star_colors)
    check_coloring_rows(sparsity, row_colors)
    assert star_num == 2
    assert row_num == 10


@pytest.mark.coloring
def test_star_what_fig_41():
    """Figure 4.1 from Gebremedhin et al. (2005), "What Color Is Your Jacobian?".

    6x6 symmetric matrix.
    SMC gets 4 colors with LargestFirst + direct decompression.
    """
    sparsity = SparsityPattern.from_dense(
        np.array(
            [
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 1, 1],
                [0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 1],
                [0, 1, 0, 0, 1, 0],
                [0, 1, 0, 1, 0, 1],
            ]
        )
    )

    colors, num_colors, _ = color_symmetric(sparsity)

    check_coloring_symmetric(sparsity, colors)
    assert num_colors <= 4


@pytest.mark.coloring
def test_star_what_fig_61():
    """Figure 6.1 from Gebremedhin et al. (2005).

    10x10 symmetric matrix.
    SMC gets 4 colors with LargestFirst + direct decompression.
    """
    sparsity = SparsityPattern.from_dense(
        np.array(
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
    )

    colors, num_colors, _ = color_symmetric(sparsity)

    check_coloring_symmetric(sparsity, colors)
    assert num_colors <= 4


@pytest.mark.coloring
@pytest.mark.parametrize(
    ("half_bw", "expected_star"),
    [(1, 3), (2, 5), (3, 7), (5, 11)],
    ids=["tridiag", "pentadiag", "bw3", "bw5"],
)
def test_star_banded(half_bw: int, expected_star: int):
    """Banded matrices have star chromatic number 2*half_bw + 1.

    From SMC theory.jl.
    Verified against SMC: the formula is ``2 * floor(rho/2) + 1``
    where ``rho = 2 * half_bw``.
    """
    sparsity = _make_banded(20, half_bw)

    colors, num_colors, _ = color_symmetric(sparsity)

    check_coloring_symmetric(sparsity, colors)
    assert num_colors == expected_star


@pytest.mark.coloring
def test_star_pentadiagonal_8x8():
    """Pentadiagonal 8x8: star coloring needs 5 colors.

    Verified against SMC.
    """
    sparsity = _make_banded(8, 2)

    colors, num_colors, _ = color_symmetric(sparsity)

    check_coloring_symmetric(sparsity, colors)
    assert num_colors == 5


@pytest.mark.coloring
def test_star_case_b_internal_vertex():
    """Regression: star coloring must forbid internal-vertex 2-colored P4s.

    Minimal counterexample (12 vertices): with LargestFirst ordering the buggy
    algorithm produces colors such that the path 0-1-4-11 has colors
    [3,0,3,0] - a 2-colored P4.  The bug was that the inner star-constraint
    check only verified ``ncc[u, cw] > 1`` (``v`` is an endpoint of the P4)
    and missed ``ncc[v, cw] > 1`` (``v`` is internal: has two neighbors
    sharing color ``cw``).
    """
    edges = [
        (0, 1),
        (0, 2),
        (0, 10),
        (1, 2),
        (1, 4),
        (1, 7),
        (1, 9),
        (1, 10),
        (3, 4),
        (4, 11),
        (5, 7),
        (5, 10),
        (6, 7),
        (6, 9),
        (7, 8),
        (7, 9),
        (7, 11),
        (8, 9),
        (8, 11),
        (9, 11),
    ]
    sparsity = _make_symmetric_reflexive_graph(12, edges)

    colors, _, _ = color_symmetric(sparsity)

    check_coloring_symmetric(sparsity, colors)


@pytest.mark.coloring
@pytest.mark.parametrize("_run", range(20))
def test_star_random_graphs(_run: int):
    """Fuzz: star coloring must be valid on random Erdos-Renyi graphs.

    The original buggy implementation passed every hand-written test case but
    failed on ~45% of random graphs in this regime because it only checked
    one of the two 2-colored-P4 cases.
    """
    rng = np.random.default_rng()
    n = int(rng.integers(8, 18))
    p = float(rng.uniform(0.2, 0.6))
    edges: list[tuple[int, int]] = [
        (i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p
    ]
    if not edges:
        pytest.skip("empty random graph")
    sparsity = _make_symmetric_reflexive_graph(n, edges)

    colors, _, _ = color_symmetric(sparsity)

    check_coloring_symmetric(sparsity, colors)


@pytest.mark.coloring
def test_star_not_square_raises():
    """Star coloring requires a square pattern."""
    sparsity = SparsityPattern.from_coo([0, 1], [0, 1], (3, 4))

    with pytest.raises(ValueError, match="square"):
        color_symmetric(sparsity)


@pytest.mark.coloring
def test_star_empty():
    """Empty pattern."""
    sparsity = SparsityPattern.from_coo([], [], (0, 0))

    colors, num_colors, _ = color_symmetric(sparsity)

    assert num_colors == 0
    assert len(colors) == 0


# Postprocessing tests


def _make_symmetric_graph_no_diagonal(
    n: int, edges: list[tuple[int, int]]
) -> SparsityPattern:
    """Symmetric adjacency pattern with no self-loops (no diagonal entries).

    Used to exercise star-coloring postprocessing, which can prune a color
    only when it is not forced-used by a diagonal nonzero.
    """
    rows = [i for i, j in edges] + [j for i, j in edges]
    cols = [j for i, j in edges] + [i for i, j in edges]
    return SparsityPattern.from_coo(rows, cols, (n, n))


@pytest.mark.coloring
def test_star_postprocessing_reduces_colors_on_c4():
    """Postprocessing on a 4-cycle (no diagonal) reduces 3 colors to 2.

    C4 has star chromatic number 3, but with LargestFirst greedy + postprocessing,
    the middle color is only assigned to vertices whose color is never used as
    a hub, so it gets pruned.
    """
    sparsity = _make_symmetric_graph_no_diagonal(4, [(0, 1), (0, 2), (1, 3), (2, 3)])

    colors_off, num_off, _ = color_symmetric(sparsity, postprocess=False)
    colors_on, num_on, _ = color_symmetric(sparsity, postprocess=True)

    check_coloring_symmetric(sparsity, colors_off)
    check_coloring_symmetric(sparsity, colors_on)
    assert num_off == 3
    assert num_on == 2
    assert num_on < num_off
    # Postprocessing introduces the neutral sentinel.
    assert (colors_on == -1).any()
    assert not (colors_off == -1).any()


@pytest.mark.coloring
def test_star_postprocessing_noop_when_full_diagonal():
    """With a full diagonal, every color is forced-used; postprocessing is a no-op."""
    sparsity = _make_banded(20, 2)

    colors_off, num_off, _ = color_symmetric(sparsity, postprocess=False)
    colors_on, num_on, _ = color_symmetric(sparsity, postprocess=True)

    check_coloring_symmetric(sparsity, colors_off)
    check_coloring_symmetric(sparsity, colors_on)
    assert num_on == num_off
    assert not (colors_on == -1).any()


@pytest.mark.coloring
def test_hessian_coloring_postprocess_flag_threaded():
    """The postprocess flag on hessian_coloring_from_sparsity reaches color_symmetric."""
    sparsity = _make_symmetric_graph_no_diagonal(4, [(0, 1), (0, 2), (1, 3), (2, 3)])

    result_off = hessian_coloring_from_sparsity(sparsity, postprocess=False)
    result_on = hessian_coloring_from_sparsity(sparsity, postprocess=True)

    assert result_off.num_colors == 3
    assert result_on.num_colors == 2
    assert result_on.num_colors < result_off.num_colors


@pytest.mark.coloring
def test_jacobian_coloring_symmetric_postprocess_flag_threaded():
    """The postprocess flag on jacobian_coloring_from_sparsity reaches color_symmetric."""
    sparsity = _make_symmetric_graph_no_diagonal(4, [(0, 1), (0, 2), (1, 3), (2, 3)])

    result_off = jacobian_coloring_from_sparsity(
        sparsity, symmetric=True, postprocess=False
    )
    result_on = jacobian_coloring_from_sparsity(
        sparsity, symmetric=True, postprocess=True
    )

    assert result_off.num_colors == 3
    assert result_on.num_colors == 2
    assert result_on.num_colors < result_off.num_colors


# Unified jacobian_coloring_from_sparsity() tests


@pytest.mark.coloring
def test_color_returns_coloring_result():
    """jacobian_coloring_from_sparsity() returns a ColoredPattern with correct fields."""
    sparsity = SparsityPattern.from_coo([0, 1, 2, 3], [0, 1, 2, 3], (4, 4))

    result = jacobian_coloring_from_sparsity(sparsity)

    assert isinstance(result, ColoredPattern)
    assert isinstance(result.num_colors, int)
    assert result.mode in ("fwd", "rev")
    assert len(result.colors) in (4, 4)  # m or n (both 4 here)


@pytest.mark.coloring
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_color_auto_picks_fwd_for_tall():
    """Auto picks fwd (column coloring) for tall-skinny patterns.

    With m=6 and n=2, column coloring needs at most 2 colors
    while row coloring may need up to 6.
    """
    # 6 rows, 2 columns — each row has one entry in each column
    rows = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    cols = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    sparsity = SparsityPattern.from_coo(rows, cols, (6, 2))

    result = jacobian_coloring_from_sparsity(sparsity)

    assert result.mode == "fwd"
    assert result.num_colors <= 2
    assert len(result.colors) == 2  # n=2


@pytest.mark.coloring
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_color_auto_picks_rev_for_wide():
    """Auto picks rev (row coloring) for wide patterns.

    With m=2 and n=6, row coloring needs at most 2 colors
    while column coloring may need up to 6.
    """
    # 2 rows, 6 columns — each column has entries in both rows
    rows = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    cols = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
    sparsity = SparsityPattern.from_coo(rows, cols, (2, 6))

    result = jacobian_coloring_from_sparsity(sparsity)

    assert result.mode == "rev"
    assert result.num_colors <= 2
    assert len(result.colors) == 2  # m=2


@pytest.mark.coloring
def test_color_force_rev():
    """jacobian_coloring_from_sparsity(sparsity, mode="rev") forces row coloring."""
    sparsity = SparsityPattern.from_coo([0, 1, 2, 3], [0, 1, 2, 3], (4, 4))

    result = jacobian_coloring_from_sparsity(sparsity, mode="rev")

    assert result.mode == "rev"
    assert len(result.colors) == 4  # m=4
    check_coloring_rows(sparsity, result.colors)


@pytest.mark.coloring
def test_color_force_fwd():
    """jacobian_coloring_from_sparsity(sparsity, mode="fwd") forces column coloring."""
    sparsity = SparsityPattern.from_coo([0, 1, 2, 3], [0, 1, 2, 3], (4, 4))

    result = jacobian_coloring_from_sparsity(sparsity, mode="fwd")

    assert result.mode == "fwd"
    assert len(result.colors) == 4  # n=4
    check_coloring_cols(sparsity, result.colors)


# jacobian_coloring / hessian_coloring tests


@pytest.mark.coloring
def test_jacobian_coloring_basic():
    """jacobian_coloring returns a correct ColoredPattern."""

    def f(x):
        return x**2

    result = jacobian_coloring(f, input_shape=(4,))

    assert isinstance(result, ColoredPattern)
    assert result.sparsity.shape == (4, 4)
    assert result.num_colors == 1  # diagonal → 1 color


@pytest.mark.coloring
def test_jacobian_coloring_mode():
    """jacobian_coloring respects the mode argument."""

    def f(x):
        return x**2

    result_rev = jacobian_coloring(f, input_shape=(3,), mode="rev")
    result_fwd = jacobian_coloring(f, input_shape=(3,), mode="fwd")

    assert result_rev.mode == "rev"
    assert result_fwd.mode == "fwd"


@pytest.mark.coloring
def test_hessian_coloring_basic():
    """hessian_coloring returns a ColoredPattern with star coloring."""

    def f(x):
        return jnp.sum(x**2)

    result = hessian_coloring(f, input_shape=(4,))

    assert isinstance(result, ColoredPattern)
    assert result.symmetric is True
    assert result.mode == "fwd_over_rev"
    assert result.sparsity.shape == (4, 4)
    # Diagonal Hessian → 1 color
    assert result.num_colors == 1


@pytest.mark.coloring
def test_hessian_coloring_coupled():
    """hessian_coloring uses star coloring for a coupled function."""

    def f(x):
        return x[0] * x[1] + x[1] * x[2] + jnp.sum(x**2)

    result = hessian_coloring(f, input_shape=(3,))

    assert isinstance(result, ColoredPattern)
    assert result.symmetric is True
    # Star coloring should use fewer colors than n for sparse Hessians
    assert result.num_colors <= 3


# _compressed_pattern tests


@pytest.mark.coloring
def test_compressed_pattern_column():
    """Column compressed pattern has shape (m, num_colors)."""
    sparsity = SparsityPattern.from_dense(
        np.array(
            [
                [1, 0, 1],
                [0, 1, 1],
                [1, 0, 0],
            ]
        )
    )
    result = jacobian_coloring_from_sparsity(sparsity, mode="fwd")
    compressed = _compressed_pattern(result)

    assert compressed.shape == (3, result.num_colors)
    # Every original row with a nonzero should appear in compressed
    dense_orig = sparsity.todense()
    dense_comp = compressed.todense()
    for i in range(3):
        has_orig = np.any(dense_orig[i] != 0)
        has_comp = np.any(dense_comp[i] != 0)
        assert has_orig == has_comp


@pytest.mark.coloring
def test_compressed_pattern_row():
    """Row compressed pattern has shape (num_colors, n)."""
    sparsity = SparsityPattern.from_dense(
        np.array(
            [
                [1, 0, 1],
                [0, 1, 1],
                [1, 0, 0],
            ]
        )
    )
    result = jacobian_coloring_from_sparsity(sparsity, mode="rev")
    compressed = _compressed_pattern(result)

    assert compressed.shape == (result.num_colors, 3)
    # Every original column with a nonzero should appear in compressed
    dense_orig = sparsity.todense()
    dense_comp = compressed.todense()
    for j in range(3):
        has_orig = np.any(dense_orig[:, j] != 0)
        has_comp = np.any(dense_comp[:, j] != 0)
        assert has_orig == has_comp


# __str__ visualization tests


@pytest.mark.coloring
def test_str_column_contains_arrow():
    """Forward mode __str__ contains → for side-by-side display."""
    sparsity = SparsityPattern.from_dense(
        np.array(
            [
                [1, 0, 1],
                [0, 1, 1],
                [1, 0, 0],
            ]
        )
    )
    result = jacobian_coloring_from_sparsity(sparsity, mode="fwd")
    s = str(result)

    assert "→" in s
    assert "●" in s


@pytest.mark.coloring
def test_str_row_contains_downarrow():
    """Row mode __str__ contains ↓ for stacked display."""
    sparsity = SparsityPattern.from_dense(
        np.array(
            [
                [1, 0, 1],
                [0, 1, 1],
                [1, 0, 0],
            ]
        )
    )
    result = jacobian_coloring_from_sparsity(sparsity, mode="rev")
    s = str(result)

    assert "↓" in s
    assert "●" in s


# hessian with coloring tests


@pytest.mark.slow
@pytest.mark.hessian
def test_hessian_with_coloring():
    """Hessian works with a pre-computed ColoredPattern."""

    def f(x):
        return jnp.sum(x**2) + x[0] * x[1]

    x = np.array([1.0, 2.0, 3.0])
    coloring = hessian_coloring(f, input_shape=x.shape)
    result = hessian_from_coloring(f, coloring)(x).todense()
    expected = jax.hessian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.hessian
def test_hessian_coloring_zero_hessian():
    """Hessian with coloring handles all-zero Hessian (nnz=0)."""

    def f(x):
        return jnp.sum(x)

    x = np.array([1.0, 2.0, 3.0])
    coloring = hessian_coloring(f, input_shape=x.shape)
    result = hessian_from_coloring(f, coloring)(x)

    assert result.shape == (3, 3)
    assert_allclose(result.todense(), np.zeros((3, 3)))


@pytest.mark.coloring
def test_str_hvp_display():
    """Symmetric ColoredPattern __str__ shows 'instead of N HVPs'."""

    def f(x):
        return jnp.sum(x**2)

    coloring = hessian_coloring(f, input_shape=(3,))
    s = str(coloring)

    assert "HVP" in s
    assert "instead of" in s
    assert "→" in s


@pytest.mark.coloring
def test_repr_coloring():
    """ColoredPattern __repr__ returns a compact single-line string."""

    def f(x):
        return x**2

    coloring = jacobian_coloring(f, input_shape=(3,))
    r = repr(coloring)

    assert "ColoredPattern" in r


@pytest.mark.coloring
def test_color_empty_pattern():
    """Coloring an empty sparsity pattern returns 0 colors."""
    sparsity = SparsityPattern.from_coo([], [], (0, 3))
    result = jacobian_coloring_from_sparsity(sparsity, mode="rev")

    assert result.num_colors == 0
    assert len(result.colors) == 0


@pytest.mark.slow
@pytest.mark.hessian
def test_hessian_star_decompression_non_unique_branch():
    """Star decompression uses fallback when a color is not unique in a column.

    With a tridiagonal Hessian and star coloring using 3 colors,
    some off-diagonal entries require the fallback decompress path
    (colors[j] in row i instead of colors[i] in column j).
    """

    def f(x):
        return jnp.sum((x[1:] - x[:-1]) ** 2)

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected = jax.hessian(f)(x)

    # Build the correct tridiagonal sparsity pattern manually
    rows, cols = [], []
    n = x.size
    for i in range(n):
        rows.append(i)
        cols.append(i)
        if i + 1 < n:
            rows.extend([i, i + 1])
            cols.extend([i + 1, i])
    sparsity = SparsityPattern.from_coo(rows, cols, (n, n))
    colors_arr, num, _ = color_symmetric(sparsity)

    # Verify star coloring reuses colors (needs only 3 for tridiagonal)
    assert num == 3

    coloring = ColoredPattern(
        sparsity,
        colors=colors_arr,
        num_colors=num,
        symmetric=True,
        mode="fwd_over_rev",
    )
    result = hessian_from_coloring(f, coloring)(x).todense()

    assert_allclose(result, expected, rtol=1e-5)


# DenseColoringWarning tests


@pytest.mark.coloring
def test_dense_jacobian_warns():
    """jacobian_coloring_from_sparsity warns when coloring is as expensive as dense."""
    rows, cols = [], []
    for i in range(4):
        for j in range(4):
            rows.append(i)
            cols.append(j)
    sparsity = SparsityPattern.from_coo(rows, cols, (4, 4))

    with pytest.warns(DenseColoringWarning, match="same as the dense case"):
        jacobian_coloring_from_sparsity(sparsity)


@pytest.mark.coloring
def test_dense_hessian_warns():
    """hessian_coloring_from_sparsity warns when coloring is as expensive as dense."""
    rows, cols = [], []
    for i in range(4):
        for j in range(4):
            rows.append(i)
            cols.append(j)
    sparsity = SparsityPattern.from_coo(rows, cols, (4, 4))

    with pytest.warns(DenseColoringWarning, match="same as the dense case"):
        hessian_coloring_from_sparsity(sparsity)


@pytest.mark.coloring
def test_dense_warning_suppressible():
    """DenseColoringWarning can be suppressed with filterwarnings."""
    rows, cols = [], []
    for i in range(4):
        for j in range(4):
            rows.append(i)
            cols.append(j)
    sparsity = SparsityPattern.from_coo(rows, cols, (4, 4))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DenseColoringWarning)
        # Should not raise any warning
        jacobian_coloring_from_sparsity(sparsity)


# Symmetric Jacobian coloring tests


@pytest.mark.coloring
def test_color_jacobian_symmetric():
    """jacobian_coloring_from_sparsity with symmetric=True returns symmetric coloring."""
    sparsity = SparsityPattern.from_coo([0, 1, 2, 3], [0, 1, 2, 3], (4, 4))

    result = jacobian_coloring_from_sparsity(sparsity, symmetric=True)

    assert result.symmetric is True
    check_coloring_symmetric(sparsity, result.colors)


@pytest.mark.coloring
def test_color_jacobian_symmetric_non_square_raises():
    """jacobian_coloring_from_sparsity with symmetric=True on non-square raises ValueError."""
    sparsity = SparsityPattern.from_coo([0, 1], [0, 1], (3, 4))

    with pytest.raises(ValueError, match="square"):
        jacobian_coloring_from_sparsity(sparsity, symmetric=True)


@pytest.mark.coloring
def test_color_jacobian_symmetric_empty_non_square_raises():
    """Empty non-square pattern with symmetric coloring raises ValueError."""
    sparsity = SparsityPattern.from_coo([], [], (3, 4))

    with pytest.raises(ValueError, match="square"):
        jacobian_coloring_from_sparsity(sparsity, symmetric=True)


@pytest.mark.coloring
def test_color_jacobian_symmetric_empty_square():
    """Empty square pattern with symmetric=True returns 0 colors."""
    sparsity = SparsityPattern.from_coo([], [], (3, 3))

    result = jacobian_coloring_from_sparsity(sparsity, symmetric=True)

    assert result.num_colors == 0
    assert result.symmetric is True
    assert len(result.colors) == 3


@pytest.mark.coloring
def test_empty_hessian_symmetric_non_square_raises():
    """Empty non-square pattern with symmetric Hessian coloring raises ValueError."""
    sparsity = SparsityPattern.from_coo([], [], (3, 4))

    with pytest.raises(ValueError, match="square"):
        hessian_coloring_from_sparsity(sparsity, symmetric=True)


# Input validation and coercion tests


@pytest.mark.coloring
def test_jacobian_coloring_from_sparsity_rejects_unsupported_type():
    """jacobian_coloring_from_sparsity raises TypeError for unsupported input."""
    with pytest.raises(TypeError, match="Expected a SparsityPattern"):
        jacobian_coloring_from_sparsity((3, 3))  # ty: ignore[invalid-argument-type]


@pytest.mark.coloring
def test_hessian_coloring_from_sparsity_rejects_unsupported_type():
    """hessian_coloring_from_sparsity raises TypeError for unsupported input."""
    with pytest.raises(TypeError, match="Expected a SparsityPattern"):
        hessian_coloring_from_sparsity((3, 3))  # ty: ignore[invalid-argument-type]


@pytest.mark.coloring
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_jacobian_coloring_from_sparsity_accepts_ndarray():
    """jacobian_coloring_from_sparsity auto-converts a numpy array."""
    dense = np.array([[1, 0], [0, 1], [1, 1]])  # (3, 2)
    result = jacobian_coloring_from_sparsity(dense)

    assert isinstance(result, ColoredPattern)
    assert result.sparsity.shape == (3, 2)
    assert result.sparsity.nnz == 4


@pytest.mark.coloring
def test_hessian_coloring_from_sparsity_accepts_ndarray():
    """hessian_coloring_from_sparsity auto-converts a numpy array."""
    dense = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
    result = hessian_coloring_from_sparsity(dense)

    assert isinstance(result, ColoredPattern)
    assert result.sparsity.shape == (3, 3)
    assert result.sparsity.nnz == 5


@pytest.mark.coloring
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_jacobian_coloring_from_sparsity_accepts_bcoo():
    """jacobian_coloring_from_sparsity auto-converts a JAX BCOO matrix."""
    dense = jnp.array([[1, 0], [0, 1], [1, 1]])
    bcoo = BCOO.fromdense(dense)
    result = jacobian_coloring_from_sparsity(bcoo)

    assert isinstance(result, ColoredPattern)
    assert result.sparsity.shape == (3, 2)
    assert result.sparsity.nnz == 4


@pytest.mark.coloring
def test_hessian_coloring_from_sparsity_accepts_bcoo():
    """hessian_coloring_from_sparsity auto-converts a JAX BCOO matrix."""
    dense = jnp.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
    bcoo = BCOO.fromdense(dense)
    result = hessian_coloring_from_sparsity(bcoo)

    assert isinstance(result, ColoredPattern)
    assert result.sparsity.shape == (3, 3)
    assert result.sparsity.nnz == 5


@pytest.mark.coloring
def test_hessian_coloring_from_sparsity_rejects_non_square():
    """hessian_coloring_from_sparsity raises ValueError for non-square pattern."""
    sparsity = SparsityPattern.from_coo([0, 1], [0, 1], (2, 3))

    with pytest.raises(ValueError, match="square"):
        hessian_coloring_from_sparsity(sparsity)


@pytest.mark.coloring
def test_hessian_coloring_from_sparsity_rejects_non_square_ndarray():
    """hessian_coloring_from_sparsity raises ValueError for non-square numpy array."""
    dense = np.array([[1, 0, 0], [0, 1, 0]])  # (2, 3)

    with pytest.raises(ValueError, match="square"):
        hessian_coloring_from_sparsity(dense)


@pytest.mark.coloring
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_color_zero_row_pattern():
    """Coloring a (0, n) pattern exercises _greedy_color with 0 vertices."""
    sparsity = SparsityPattern.from_coo([0], [0], (1, 3))

    # Force row coloring on a pattern where m=1 → single vertex
    result = jacobian_coloring_from_sparsity(sparsity, mode="rev")
    assert result.num_colors == 1

    # Now test with m=0
    sparsity_zero = SparsityPattern.from_coo([], [], (0, 3))
    result_zero = jacobian_coloring_from_sparsity(sparsity_zero, mode="rev")
    assert result_zero.num_colors == 0
    assert len(result_zero.colors) == 0


# Mode-handling tests


@pytest.mark.coloring
def test_empty_jacobian_default_mode_is_fwd():
    """Empty non-square Jacobian with no mode defaults to ``fwd`` and sizes colors to n.

    The fallback picks fwd (JVPs cheaper than VJPs) and allocates a length-n
    sentinel color vector so downstream decompression can ``colors[col]`` lookup
    without a mode-dependent branch.
    """
    sparsity = SparsityPattern.from_coo([], [], (3, 4))

    result = jacobian_coloring_from_sparsity(sparsity)

    assert result.mode == "fwd"
    assert result.num_colors == 0
    assert len(result.colors) == 4  # n=4, not m=3
    assert np.all(result.colors == -1)


@pytest.mark.coloring
def test_hessian_coloring_explicit_mode_roundtrip():
    """An explicit Hessian mode threads through coloring to a correct HVP decompression."""

    def f(x):
        return jnp.sum(x**2) + x[0] * x[1]

    x = np.array([1.0, 2.0, 3.0])
    coloring = hessian_coloring(f, input_shape=x.shape, mode="rev_over_fwd")

    assert coloring.mode == "rev_over_fwd"
    result = hessian_from_coloring(f, coloring)(x).todense()
    expected = jax.hessian(f)(x)
    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.coloring
def test_hessian_coloring_invalid_mode_raises():
    """hessian_coloring_from_sparsity rejects an unknown mode string."""
    sparsity = SparsityPattern.from_coo([0, 1], [0, 1], (2, 2))

    with pytest.raises(ValueError, match="Unknown mode"):
        hessian_coloring_from_sparsity(sparsity, mode="bogus")  # ty: ignore[invalid-argument-type]


@pytest.mark.coloring
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_hessian_coloring_non_symmetric_column_roundtrip():
    """symmetric=False falls back to column coloring and still recovers the Hessian."""

    def f(x):
        return x[0] * x[1] + x[1] * x[2] + jnp.sum(x**2)

    x = np.array([1.0, 2.0, 3.0])
    coloring = hessian_coloring(f, input_shape=x.shape, symmetric=False)

    assert coloring.symmetric is False
    assert coloring.star_set is None
    assert len(coloring.colors) == 3
    check_coloring_cols(coloring.sparsity, coloring.colors)

    result = hessian_from_coloring(f, coloring)(x).todense()
    expected = jax.hessian(f)(x)
    assert_allclose(result, expected, rtol=1e-5)


# forced_colors tests


@pytest.mark.coloring
def test_color_symmetric_forced_colors_overrides_greedy_choice():
    """A valid forced coloring overrides what greedy would pick on its own.

    Greedy picks a 2-coloring on path 0-1-2 (star chromatic number is 2).
    Forcing a 3-coloring returns that distinct assignment verbatim,
    demonstrating that forced colors are used instead of recomputed, and
    the companion star set is rebuilt around the forced colors.
    """
    sparsity = _make_symmetric_graph_no_diagonal(3, [(0, 1), (1, 2)])
    _, greedy_num, _ = color_symmetric(sparsity)
    assert greedy_num == 2  # sanity: baseline differs from forced

    forced = np.array([0, 1, 2], dtype=np.int32)
    colors, num, star_set = color_symmetric(sparsity, forced_colors=forced)

    check_coloring_symmetric(sparsity, colors)
    np.testing.assert_array_equal(colors, forced)
    assert num == 3
    # With 3 distinct colors every edge is a trivial star (no shared-color
    # neighbor to absorb into). The star set must still cover both edges.
    assert set(star_set.edge_index) == {(0, 1), (1, 2)}


@pytest.mark.coloring
def test_color_symmetric_forced_colors_accepts_list():
    """color_symmetric accepts a plain Python list for forced_colors."""
    sparsity = _make_symmetric_graph_no_diagonal(3, [(0, 1), (1, 2)])

    colors, _, _ = color_symmetric(sparsity, forced_colors=[0, 1, 0])

    check_coloring_symmetric(sparsity, colors)
    np.testing.assert_array_equal(colors, np.array([0, 1, 0], dtype=np.int32))


@pytest.mark.coloring
def test_color_symmetric_forced_colors_wrong_shape_raises():
    """forced_colors with wrong shape raises ValueError."""
    sparsity = _make_symmetric_graph_no_diagonal(3, [(0, 1), (1, 2)])

    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        color_symmetric(sparsity, forced_colors=np.array([0, 1], dtype=np.int32))


@pytest.mark.coloring
def test_color_symmetric_forced_colors_negative_raises():
    """forced_colors with negative values raises ValueError."""
    sparsity = _make_symmetric_graph_no_diagonal(3, [(0, 1), (1, 2)])

    with pytest.raises(ValueError, match="non-negative"):
        color_symmetric(sparsity, forced_colors=np.array([0, -1, 1], dtype=np.int32))


@pytest.mark.coloring
def test_color_symmetric_forced_colors_distance1_violation_raises():
    """Adjacent vertices sharing a forced color violate the star constraint."""
    sparsity = _make_symmetric_graph_no_diagonal(2, [(0, 1)])

    with pytest.raises(InvalidColoringError, match="violates a star-coloring"):
        color_symmetric(sparsity, forced_colors=np.array([0, 0], dtype=np.int32))


# StarSet.hub_vertex tests


@pytest.mark.coloring
def test_star_set_hub_vertex_resolved():
    """hub_vertex returns the shared endpoint as hub of a resolved 2-edge star.

    On path 0-1-2 the greedy algorithm merges both edges into one star
    whose hub is vertex 1 (the only vertex with >1 neighbor).
    """
    sparsity = _make_symmetric_graph_no_diagonal(3, [(0, 1), (1, 2)])

    _, _, star_set = color_symmetric(sparsity)

    assert star_set.hub_vertex(0, 1) == 1
    assert star_set.hub_vertex(1, 2) == 1
    # Argument order does not affect lookup.
    assert star_set.hub_vertex(2, 1) == 1
    assert star_set.hub_vertex(1, 0) == 1


@pytest.mark.coloring
def test_star_set_hub_vertex_unresolved_trivial_star():
    """hub_vertex decodes the default endpoint for an unresolved trivial star.

    Trivial stars store the default hub as ``-(v + 1)``;
    ``hub_vertex`` must reverse that encoding regardless of argument order.
    """
    star_set = StarSet(
        star=np.array([0], dtype=np.int32),
        hub=np.array([-2], dtype=np.int32),  # encodes default endpoint v=1
        edge_index={(0, 1): 0},
    )

    assert star_set.hub_vertex(0, 1) == 1
    assert star_set.hub_vertex(1, 0) == 1


# Postprocessing: trivial-star hub flip


@pytest.mark.coloring
def test_postprocess_trivial_star_marks_default_hub_color_used():
    """Trivial stars with fresh spoke colors mark the default hub's color used.

    Two disjoint edges with no diagonal entries form two trivial stars.
    Greedy assigns color 0 to both spokes (0 and 2) and color 1 to both
    default hubs (1 and 3). During postprocessing neither spoke color has
    been marked used by a non-trivial star, so the default-hub branch
    records color 1 as used; color 0 gets pruned, leaving the spokes with
    the neutral ``-1`` sentinel and compacting color 1 down to 0.
    """
    sparsity = _make_symmetric_graph_no_diagonal(4, [(0, 1), (2, 3)])

    colors_on, num_on, _ = color_symmetric(sparsity, postprocess=True)

    check_coloring_symmetric(sparsity, colors_on)
    # Hubs keep an active color; spokes are pruned to the neutral sentinel.
    assert num_on == 1
    np.testing.assert_array_equal(colors_on, np.array([-1, 0, -1, 0], dtype=np.int32))


@pytest.mark.coloring
def test_postprocess_trivial_star_flips_hub_to_keep_used_color():
    """A trivial star flips its hub when the spoke already carries a used color.

    Graph: path 0-1-2 joined to a disjoint edge 3-4.
    Star-coloring with LargestFirst visits vertex 1 first (highest degree),
    colors it 0, then colors 0, 2, 3, 4 with color 1. Vertex 1 becomes
    the hub of the path star, so color 0 is marked used. The trivial
    star on edge (3, 4) defaults to hub=4 (the max endpoint, color 1),
    but its spoke vertex 3 already has color 1 — so the flip branch
    reassigns the hub to 3, and color 1 remains "used" exactly once.
    """
    sparsity = _make_symmetric_graph_no_diagonal(5, [(0, 1), (1, 2), (3, 4)])

    colors_off, num_off, star_off = color_symmetric(sparsity, postprocess=False)
    colors_on, num_on, star_on = color_symmetric(sparsity, postprocess=True)

    check_coloring_symmetric(sparsity, colors_off)
    check_coloring_symmetric(sparsity, colors_on)
    # Without postprocess, the trivial-star edge (3, 4) has an unresolved hub.
    assert star_off.hub_vertex(3, 4) == 4  # default = max endpoint
    # Postprocess flips the hub to the spoke whose color is already used.
    assert star_on.hub_vertex(3, 4) == 3
    # Flipping collapses the color count from 2 down to 1.
    assert num_off == 2
    assert num_on == 1
