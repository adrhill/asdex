"""Check asdex's coloring validators against SparseMatrixColorings.jl.

``check_coloring_cols`` / ``check_coloring_rows`` decide structural
orthogonality, which SMC exposes as ``structurally_orthogonal_columns``.
``check_coloring_symmetric`` decides whether a coloring is a star coloring,
which is equivalent to the symmetric orthogonality decided by SMC's
``symmetrically_orthogonal_columns`` — but only when the diagonal is fully
non-zero, so the symmetric comparisons fill the diagonal in
(see ``_matrices.with_diagonal``).

The hard-coded cases come from ``test/check.jl`` in SparseMatrixColorings.jl,
converted to 0-based colors.
"""

import numpy as np
import pytest

from asdex import (
    SparsityPattern,
    check_coloring_cols,
    check_coloring_rows,
    check_coloring_symmetric,
)
from asdex.coloring import InvalidColoringError, color_symmetric
from tests.smc._matrices import (
    SAMPLES,
    STRUCTURED_SYMMETRIC_MATRICES,
    SYMMETRIC_PARAMS,
    random_matrix,
    random_symmetric_matrix,
    with_diagonal,
)

pytestmark = [pytest.mark.smc, pytest.mark.coloring]

# From the "Structurally orthogonal columns" testset of SMC's test/check.jl.
_TRIANGLE = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 1],
    ]
)

# From the "Symmetrically orthogonal" testset of SMC's test/check.jl,
# shifted from SMC's 1-based colors to asdex's 0-based colors.
_WHAT_FIG_41_VALID = [0, 1, 0, 2, 0, 0]
_WHAT_FIG_41_INVALID = [0, 2, 0, 2, 0, 0]
_EFFICIENT_FIG_1_VALID = [0, 1, 0, 2, 0, 3, 2, 4, 0, 1]
_EFFICIENT_FIG_1_INVALID = [
    [0, 1, 0, 2, 0, 3, 2, 3, 0, 1],
    [0, 1, 0, 2, 0, 3, 1, 4, 0, 1],
    [0, 1, 0, 3, 0, 3, 2, 4, 0, 1],
]


# Shapes, sizes and densities for the randomized checker comparisons.
# Sparse patterns make most colorings valid and dense ones make most invalid,
# so sweeping the density is what gets both verdicts in front of the checkers;
# test_random_colorings_cover_both_verdicts asserts that it actually works.
_SHAPES = [(4, 6), (6, 4), (8, 8), (12, 5)]
_SIZES = [4, 6, 8, 10]
_DENSITIES = [0.1, 0.3, 0.5, 0.8]

# Random colorings drawn per color count, for each pattern.
_DRAWS_PER_COLOR_COUNT = 3

# Distinct seed offsets so the three checkers see independent colorings.
_KIND_SEEDS = {"cols": 1, "rows": 2, "symmetric": 3}


def _is_valid(check, sparsity, colors) -> bool:
    """Run a checker and report validity as a boolean instead of an exception."""
    try:
        check(sparsity, np.asarray(colors, dtype=np.int32))
    except InvalidColoringError:
        return False
    return True


def _random_colorings(kind: str, size: int, m: int, n: int, p: float):
    """Deterministic batch of random colorings, spanning every color count.

    Shared by the agreement tests and by
    ``test_random_colorings_cover_both_verdicts``,
    so the coverage claim is about exactly the colorings that get compared.
    """
    rng = np.random.default_rng([_KIND_SEEDS[kind], size, m, n, round(p * 1000)])
    for num_colors in range(1, size + 1):
        for _ in range(_DRAWS_PER_COLOR_COUNT):
            yield rng.integers(0, num_colors, size=size).astype(np.int32)


# Hard-coded cases from SMC's test suite


@pytest.mark.parametrize("colors", [[0, 1, 2], [0, 1, 0], [0, 0, 1]])
def test_check_coloring_cols_accepts_smc_cases(smc, colors):
    """Column colorings that SMC calls structurally orthogonal are accepted."""
    sparsity = SparsityPattern.from_dense(_TRIANGLE)
    check_coloring_cols(sparsity, np.asarray(colors, dtype=np.int32))
    assert smc.orthogonal_cols(_TRIANGLE, colors)


def test_check_coloring_cols_rejects_smc_case(smc):
    """Columns 2 and 3 share color 1 and both hit row 3, so the coloring is invalid."""
    sparsity = SparsityPattern.from_dense(_TRIANGLE)
    colors = np.array([0, 1, 1], dtype=np.int32)
    with pytest.raises(InvalidColoringError, match="Invalid column coloring"):
        check_coloring_cols(sparsity, colors)
    assert not smc.orthogonal_cols(_TRIANGLE, colors)


@pytest.mark.parametrize("colors", [[0, 1, 2], [0, 1, 0], [0, 0, 1]])
def test_check_coloring_rows_accepts_smc_cases(smc, colors):
    """Row colorings of the transposed pattern are accepted, mirroring SMC."""
    sparsity = SparsityPattern.from_dense(_TRIANGLE.T)
    check_coloring_rows(sparsity, np.asarray(colors, dtype=np.int32))
    assert smc.orthogonal_rows(_TRIANGLE.T, colors)


def test_check_coloring_rows_rejects_smc_case(smc):
    """Rows 2 and 3 of the transposed pattern share a color and a column."""
    sparsity = SparsityPattern.from_dense(_TRIANGLE.T)
    colors = np.array([0, 1, 1], dtype=np.int32)
    with pytest.raises(InvalidColoringError, match="Invalid row coloring"):
        check_coloring_rows(sparsity, colors)
    assert not smc.orthogonal_rows(_TRIANGLE.T, colors)


@pytest.mark.parametrize(
    ("name", "colors"),
    [
        ("what_fig_41", _WHAT_FIG_41_VALID),
        ("efficient_fig_1", _EFFICIENT_FIG_1_VALID),
    ],
)
def test_check_coloring_symmetric_accepts_smc_cases(smc, name, colors):
    """The published star colorings of the two SMC example matrices are accepted."""
    matrix = STRUCTURED_SYMMETRIC_MATRICES[name]
    check_coloring_symmetric(
        SparsityPattern.from_dense(matrix), np.asarray(colors, dtype=np.int32)
    )
    assert smc.orthogonal_symmetric(matrix, colors)


@pytest.mark.parametrize(
    ("name", "colors"),
    [
        ("what_fig_41", _WHAT_FIG_41_INVALID),
        *[("efficient_fig_1", c) for c in _EFFICIENT_FIG_1_INVALID],
    ],
)
def test_check_coloring_symmetric_rejects_smc_cases(smc, name, colors):
    """Colorings that SMC rejects as not symmetrically orthogonal are rejected."""
    matrix = STRUCTURED_SYMMETRIC_MATRICES[name]
    with pytest.raises(InvalidColoringError, match="Invalid star coloring"):
        check_coloring_symmetric(
            SparsityPattern.from_dense(matrix), np.asarray(colors, dtype=np.int32)
        )
    assert not smc.orthogonal_symmetric(matrix, colors)


# Randomized agreement with SMC's validators


@pytest.mark.parametrize("p", _DENSITIES)
@pytest.mark.parametrize(("m", "n"), _SHAPES)
def test_check_coloring_cols_agrees_with_smc(smc, m, n, p):
    """Asdex and SMC agree on every random column coloring of a random pattern."""
    matrix = random_matrix(m, n, p, sample=0)
    sparsity = SparsityPattern.from_dense(matrix)
    for colors in _random_colorings("cols", n, m, n, p):
        assert _is_valid(check_coloring_cols, sparsity, colors) == smc.orthogonal_cols(
            matrix, colors
        )


@pytest.mark.parametrize("p", _DENSITIES)
@pytest.mark.parametrize(("m", "n"), _SHAPES)
def test_check_coloring_rows_agrees_with_smc(smc, m, n, p):
    """Asdex and SMC agree on every random row coloring of a random pattern."""
    matrix = random_matrix(m, n, p, sample=0)
    sparsity = SparsityPattern.from_dense(matrix)
    for colors in _random_colorings("rows", m, m, n, p):
        assert _is_valid(check_coloring_rows, sparsity, colors) == smc.orthogonal_rows(
            matrix, colors
        )


@pytest.mark.parametrize("p", _DENSITIES)
@pytest.mark.parametrize("n", _SIZES)
def test_check_coloring_symmetric_agrees_with_smc(smc, n, p):
    """Asdex's star check and SMC's symmetric-orthogonality check agree.

    The equivalence between the two notions requires a non-zero diagonal,
    which is also what a Hessian sparsity pattern looks like in practice.
    """
    matrix = random_symmetric_matrix(n, p, sample=0, diagonal=True)
    sparsity = SparsityPattern.from_dense(matrix)
    for colors in _random_colorings("symmetric", n, n, n, p):
        assert _is_valid(
            check_coloring_symmetric, sparsity, colors
        ) == smc.orthogonal_symmetric(matrix, colors)


def test_random_colorings_cover_both_verdicts():
    """The randomized sweeps really do produce both valid and invalid colorings.

    Without this the agreement tests above could pass vacuously,
    by only ever drawing colorings that both implementations accept.
    Needs no Julia: it replays the same seeded colorings through asdex alone.
    """
    verdicts: dict[str, set[bool]] = {"cols": set(), "rows": set(), "symmetric": set()}
    for p in _DENSITIES:
        for m, n in _SHAPES:
            sparsity = SparsityPattern.from_dense(random_matrix(m, n, p, sample=0))
            for colors in _random_colorings("cols", n, m, n, p):
                verdicts["cols"].add(_is_valid(check_coloring_cols, sparsity, colors))
            for colors in _random_colorings("rows", m, m, n, p):
                verdicts["rows"].add(_is_valid(check_coloring_rows, sparsity, colors))
        for n in _SIZES:
            matrix = random_symmetric_matrix(n, p, sample=0, diagonal=True)
            sparsity = SparsityPattern.from_dense(matrix)
            for colors in _random_colorings("symmetric", n, n, n, p):
                verdicts["symmetric"].add(
                    _is_valid(check_coloring_symmetric, sparsity, colors)
                )
    assert verdicts == {
        "cols": {False, True},
        "rows": {False, True},
        "symmetric": {False, True},
    }


# Greedy colorings must pass both sides' validators


@pytest.mark.parametrize("sample", SAMPLES)
@pytest.mark.parametrize(("n", "p"), SYMMETRIC_PARAMS)
def test_greedy_symmetric_coloring_is_symmetrically_orthogonal(smc, n, p, sample):
    """Asdex's star coloring is accepted by SMC's symmetric-orthogonality check."""
    matrix = random_symmetric_matrix(n, p, sample, diagonal=True)
    sparsity = SparsityPattern.from_dense(matrix)
    colors, _, _ = color_symmetric(sparsity)
    check_coloring_symmetric(sparsity, colors)
    assert smc.orthogonal_symmetric(matrix, colors)


@pytest.mark.parametrize("name", list(STRUCTURED_SYMMETRIC_MATRICES))
def test_greedy_structured_coloring_is_symmetrically_orthogonal(smc, name):
    """The same holds on the deterministic patterns, once a diagonal is added."""
    matrix = with_diagonal(STRUCTURED_SYMMETRIC_MATRICES[name])
    sparsity = SparsityPattern.from_dense(matrix)
    colors, _, _ = color_symmetric(sparsity)
    check_coloring_symmetric(sparsity, colors)
    assert smc.orthogonal_symmetric(matrix, colors)
