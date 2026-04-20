"""Tests for SparsityPattern data structure."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import ShapeDtypeStruct
from jax.experimental.sparse import BCOO

from asdex import ColoredPattern, SparsityPattern, jacobian_sparsity
from asdex._display import _render_braille, _render_dots


class TestValidation:
    """Test input validation."""

    def test_mismatched_rows_cols_raises(self):
        """Rows and cols with different lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            SparsityPattern.from_coo([0, 1], [0], (2, 2))


class TestConstruction:
    """Test SparsityPattern construction methods."""

    def test_from_coo(self):
        """Basic construction from row/col arrays."""
        rows = [0, 0, 1, 2]
        cols = [0, 1, 1, 2]
        sparsity = SparsityPattern.from_coo(rows, cols, (3, 3))

        assert sparsity.shape == (3, 3)
        assert sparsity.nnz == 4
        assert sparsity.m == 3
        assert sparsity.n == 3
        np.testing.assert_array_equal(sparsity.rows, [0, 0, 1, 2])
        np.testing.assert_array_equal(sparsity.cols, [0, 1, 1, 2])

    def test_from_coo_empty(self):
        """Construction with no non-zeros."""
        sparsity = SparsityPattern.from_coo([], [], (3, 4))

        assert sparsity.shape == (3, 4)
        assert sparsity.nnz == 0
        assert sparsity.m == 3
        assert sparsity.n == 4

    def test_from_bcoo_roundtrip(self):
        """Convert from BCOO and back."""
        # Create a BCOO matrix
        data = jnp.array([1, 1, 1])
        indices = jnp.array([[0, 0], [1, 1], [2, 2]])
        bcoo = BCOO((data, indices), shape=(3, 3))

        # Convert to SparsityPattern
        sparsity = SparsityPattern.from_bcoo(bcoo)
        assert sparsity.shape == (3, 3)
        assert sparsity.nnz == 3

        # Convert back to BCOO
        bcoo2 = sparsity.to_bcoo()
        assert bcoo2.shape == (3, 3)
        np.testing.assert_array_equal(bcoo2.todense(), bcoo.todense())

    def test_from_bcoo_empty(self):
        """Convert empty BCOO to SparsityPattern."""
        data = jnp.array([])
        indices = jnp.zeros((0, 2), dtype=jnp.int32)
        bcoo = BCOO((data, indices), shape=(3, 4))

        sparsity = SparsityPattern.from_bcoo(bcoo)
        assert sparsity.shape == (3, 4)
        assert sparsity.nnz == 0

    def test_from_dense(self):
        """Construction from dense matrix."""
        dense = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        sparsity = SparsityPattern.from_dense(dense)

        assert sparsity.shape == (3, 3)
        assert sparsity.nnz == 5
        np.testing.assert_array_equal(sparsity.todense(), (dense != 0).astype(np.int8))


class TestConversion:
    """Test conversion methods."""

    def test_todense(self):
        """Convert to dense numpy array."""
        sparsity = SparsityPattern.from_coo([0, 1, 2], [0, 1, 2], (3, 3))
        dense = sparsity.todense()

        expected = np.eye(3, dtype=np.int8)
        np.testing.assert_array_equal(dense, expected)

    def test_todense_empty(self):
        """Todense with no non-zeros."""
        sparsity = SparsityPattern.from_coo([], [], (2, 3))
        dense = sparsity.todense()

        expected = np.zeros((2, 3), dtype=np.int8)
        np.testing.assert_array_equal(dense, expected)

    def test_to_bcoo_with_data(self):
        """to_bcoo with custom data values."""
        sparsity = SparsityPattern.from_coo([0, 1, 2], [0, 1, 2], (3, 3))
        data = jnp.array([2.0, 3.0, 4.0])
        bcoo = sparsity.to_bcoo(data=data)

        expected = np.diag([2.0, 3.0, 4.0])
        np.testing.assert_array_equal(bcoo.todense(), expected)

    def test_to_bcoo_default_data(self):
        """to_bcoo uses 1s by default."""
        sparsity = SparsityPattern.from_coo([0, 1], [0, 1], (2, 2))
        bcoo = sparsity.to_bcoo()

        np.testing.assert_array_equal(bcoo.todense(), np.eye(2))

    def test_to_bcoo_empty(self):
        """to_bcoo with empty pattern produces zero matrix."""
        sparsity = SparsityPattern.from_coo([], [], (3, 4))
        bcoo = sparsity.to_bcoo()

        assert bcoo.shape == (3, 4)
        np.testing.assert_array_equal(bcoo.todense(), np.zeros((3, 4)))

    def test_to_bcoo_empty_with_data(self):
        """to_bcoo with empty pattern and custom data."""
        sparsity = SparsityPattern.from_coo([], [], (2, 2))
        data = jnp.array([])
        bcoo = sparsity.to_bcoo(data=data)

        assert bcoo.shape == (2, 2)
        np.testing.assert_array_equal(bcoo.todense(), np.zeros((2, 2)))


class TestProperties:
    """Test computed properties."""

    def test_density(self):
        """Density calculation."""
        # 2 non-zeros in 3x4 = 12 elements
        sparsity = SparsityPattern.from_coo([0, 1], [0, 1], (3, 4))
        assert sparsity.density == pytest.approx(2 / 12)

    def test_density_empty(self):
        """Density of empty pattern."""
        sparsity = SparsityPattern.from_coo([], [], (3, 4))
        assert sparsity.density == 0.0

    def test_density_zero_size(self):
        """Density with zero-size matrix."""
        sparsity = SparsityPattern.from_coo([], [], (0, 4))
        assert sparsity.density == 0.0

    def test_col_to_rows(self):
        """col_to_rows mapping."""
        # Pattern: row 0 has cols 0,1; row 1 has col 1; row 2 has col 2
        sparsity = SparsityPattern.from_coo([0, 0, 1, 2], [0, 1, 1, 2], (3, 3))

        col_to_rows = sparsity.col_to_rows
        assert col_to_rows == {0: [0], 1: [0, 1], 2: [2]}

    def test_col_to_rows_caching(self):
        """col_to_rows is cached."""
        sparsity = SparsityPattern.from_coo([0, 1], [0, 1], (2, 2))

        # Access twice - should be same object
        first = sparsity.col_to_rows
        second = sparsity.col_to_rows
        assert first is second


class TestVisualization:
    """Test visualization (dots for small, braille for large)."""

    def test_small_matrix_uses_dots(self):
        """Small matrices use dot display (●/⋅)."""
        sparsity = SparsityPattern.from_coo([0, 1, 2], [0, 1, 2], (3, 3))
        s = str(sparsity)

        # Should have header line
        assert "SparsityPattern" in s
        assert "3×3" in s
        assert "nnz=3" in s
        # Should have dots, not braille
        assert "●" in s
        assert "⋅" in s

    def test_large_matrix_uses_braille(self):
        """Large matrices use braille display."""
        # Create 20x50 pattern (exceeds thresholds)
        rows = list(range(20))
        cols = list(range(20))
        sparsity = SparsityPattern.from_coo(rows, cols, (20, 50))
        s = str(sparsity)

        # Should have braille characters (Unicode block starting at 0x2800)
        assert any(ord(c) >= 0x2800 and ord(c) < 0x2900 for c in s)
        # Should have Julia-style bracket borders
        assert "⎡" in s
        assert "⎦" in s

    def test_repr_compact(self):
        """__repr__ is compact."""
        sparsity = SparsityPattern.from_coo([0, 1], [0, 1], (10, 20))
        r = repr(sparsity)

        assert "SparsityPattern" in r
        assert "shape=(10, 20)" in r
        assert "nnz=2" in r
        # Should be single line
        assert "\n" not in r

    def test_render_dots_empty_matrix(self):
        """Dot rendering of empty matrix."""
        sparsity = SparsityPattern.from_coo([], [], (0, 0))
        assert _render_dots(sparsity) == "(empty)"

    def test_render_dots_small_diagonal(self):
        """Dot rendering of small diagonal pattern."""
        sparsity = SparsityPattern.from_coo([0, 1, 2], [0, 1, 2], (3, 3))
        dots = _render_dots(sparsity)

        # Should show diagonal pattern
        lines = dots.split("\n")
        assert len(lines) == 3
        assert "●" in lines[0]
        assert "⋅" in lines[0]

    def test_braille_empty_matrix(self):
        """Braille rendering of empty matrix."""
        sparsity = SparsityPattern.from_coo([], [], (0, 0))
        assert _render_braille(sparsity) == "(empty)"

    def test_braille_large_matrix_downsamples(self):
        """Large matrices are downsampled in braille."""
        # Create 100x100 diagonal
        rows = list(range(100))
        cols = list(range(100))
        sparsity = SparsityPattern.from_coo(rows, cols, (100, 100))

        braille = _render_braille(sparsity, max_height=10, max_width=20)
        lines = braille.split("\n")

        # Should be within limits
        assert len(lines) <= 10
        assert all(len(line) <= 20 for line in lines)

    def test_braille_preserves_aspect_ratio(self):
        """Tall, slim matrices produce narrow braille output."""
        # 20000×20 should not render as a nearly-square grid.
        # With uniform scaling, the width should be much smaller than max_width.
        rows = list(range(0, 20000, 100))
        cols = [i % 20 for i in range(len(rows))]
        sparsity = SparsityPattern.from_coo(rows, cols, (20000, 20))

        braille = _render_braille(sparsity, max_height=20, max_width=40)
        lines = braille.split("\n")

        # Height should use most of the available space
        assert len(lines) >= 15
        # Width should be narrow (2 braille chars), not stretched to 40
        assert all(len(line) <= 5 for line in lines)

    def test_large_zero_dim_matrix_str(self):
        """Large matrix with zero dimension uses braille "(empty)" fallback in __str__.

        When m or n is 0 but exceeds small-matrix thresholds,
        braille returns "(empty)" and __str__ uses it directly.
        """
        # n=50 exceeds _SMALL_COLS=40, forcing braille path; m=0 triggers "(empty)"
        sparsity = SparsityPattern.from_coo([], [], (0, 50))
        s = str(sparsity)

        assert "SparsityPattern" in s
        assert "nnz=0" in s
        assert "(empty)" in s


class TestIntegration:
    """Integration tests with detection pipeline."""

    def test_jacobian_sparsity_returns_pattern(self):
        """jacobian_sparsity returns SparsityPattern."""

        def f(x):
            return jnp.array([x[0] * x[1], x[1] + x[2], x[2]])

        result = jacobian_sparsity(f, input_shape=(3,))

        assert isinstance(result, SparsityPattern)
        assert result.shape == (3, 3)

        # Check sparsity pattern is correct
        expected = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
        np.testing.assert_array_equal(result.todense(), expected)

    def test_existing_tests_still_work(self):
        """Existing test patterns like .todense().astype(int) work."""

        def f(x):
            return x**2

        result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
        expected = np.eye(3, dtype=int)
        np.testing.assert_array_equal(result, expected)

    def test_print_sparsity(self):
        """Manual verification helper - prints sparsity pattern."""

        def f(x):
            return jnp.array([x[0] * x[1], x[1] + x[2], x[2]])

        sparsity = jacobian_sparsity(f, input_shape=(3,))
        # This should print nicely with braille
        output = str(sparsity)
        assert len(output) > 0


# --- Save/Load tests ---


def test_save_load_sparsity_roundtrip(tmp_path):
    """SparsityPattern survives a save/load roundtrip."""
    original = SparsityPattern.from_coo([0, 0, 1, 2], [0, 1, 1, 2], (3, 3))
    path = tmp_path / "pattern.npz"
    original.save(path)

    loaded = SparsityPattern.load(path)

    assert loaded.shape == original.shape
    assert loaded.leaf_shapes == original.leaf_shapes
    assert loaded.nnz == original.nnz
    np.testing.assert_array_equal(loaded.rows, original.rows)
    np.testing.assert_array_equal(loaded.cols, original.cols)


@pytest.mark.parametrize(
    ("symmetric", "mode"),
    [
        (False, "fwd"),
        (False, "rev"),
        (True, "fwd_over_rev"),
        (True, "rev_over_fwd"),
        (True, "rev_over_rev"),
    ],
    ids=["fwd", "rev", "fwd_over_rev", "rev_over_fwd", "rev_over_rev"],
)
def test_save_load_colored_roundtrip(tmp_path, symmetric, mode):
    """ColoredPattern survives a save/load roundtrip for each mode."""
    sparsity = SparsityPattern.from_coo([0, 1, 2], [0, 1, 2], (3, 3))
    # One color per row/col for a diagonal pattern.
    colors = np.array([0, 1, 2], dtype=np.int32)
    original = ColoredPattern(
        sparsity=sparsity,
        colors=colors,
        num_colors=3,
        symmetric=symmetric,
        mode=mode,
    )
    path = tmp_path / "colored.npz"
    original.save(path)

    loaded = ColoredPattern.load(path)

    assert loaded.sparsity.shape == original.sparsity.shape
    assert loaded.sparsity.leaf_shapes == original.sparsity.leaf_shapes
    np.testing.assert_array_equal(loaded.sparsity.rows, original.sparsity.rows)
    np.testing.assert_array_equal(loaded.sparsity.cols, original.sparsity.cols)
    np.testing.assert_array_equal(loaded.colors, original.colors)
    assert loaded.num_colors == original.num_colors
    assert loaded.symmetric == original.symmetric
    assert loaded.mode == original.mode


def test_save_load_sparsity_empty(tmp_path):
    """Empty SparsityPattern survives a save/load roundtrip."""
    original = SparsityPattern.from_coo([], [], (3, 4))
    path = tmp_path / "empty.npz"
    original.save(path)

    loaded = SparsityPattern.load(path)

    assert loaded.shape == (3, 4)
    assert loaded.nnz == 0
    assert loaded.leaf_shapes == [(4,)]


def test_save_load_sparsity_non_default_input_shape(tmp_path):
    """SparsityPattern with multidimensional input aval roundtrips correctly."""
    aval = ShapeDtypeStruct((2, 3), jnp.float_)
    original = SparsityPattern.from_coo([0, 1], [0, 1], (2, 6), input_avals=(aval,))
    path = tmp_path / "nd.npz"
    original.save(path)

    loaded = SparsityPattern.load(path)

    assert loaded.leaf_shapes == [(2, 3)]
    assert loaded.shape == (2, 6)


def test_load_colored_pattern_invalid_mode(tmp_path):
    """ColoredPattern.load raises ValueError when the saved mode is invalid."""
    sparsity = SparsityPattern.from_coo([0, 1, 2], [0, 1, 2], (3, 3))
    coloring = ColoredPattern(
        sparsity=sparsity,
        colors=np.array([0, 0, 0], dtype=np.int32),
        num_colors=1,
        symmetric=False,
        mode="fwd",
    )
    path = tmp_path / "bad_mode.npz"
    coloring.save(path)

    # Corrupt the mode field by re-saving with an invalid mode string
    data = dict(np.load(path))
    data["mode"] = np.array("bogus")
    np.savez(path, **data)

    with pytest.raises(ValueError, match="Unknown mode"):
        ColoredPattern.load(path)
