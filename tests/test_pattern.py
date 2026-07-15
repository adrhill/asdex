"""Tests for SparsityPattern data structure."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import ShapeDtypeStruct
from jax.core import Tracer
from jax.experimental.sparse import BCOO

import asdex
from asdex import ColoredPattern, SparsityPattern, jacobian_sparsity
from asdex._check import _allclose_pytree
from asdex._display import _render_braille, _render_dots
from asdex.coloring import StarSet


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

    def test_to_bcoo_sets_structure_flags_for_sorted_unique(self):
        """Row-major sorted patterns without duplicates get the BCOO structure flags.

        The flags let downstream sparse ops skip sorting and deduplication.
        """
        sparsity = SparsityPattern.from_coo([0, 0, 1], [0, 2, 1], (2, 3))
        bcoo = sparsity.to_bcoo()

        assert bcoo.indices_sorted
        assert bcoo.unique_indices

    def test_to_bcoo_no_structure_flags_for_unsorted(self):
        """User patterns with unsorted entries do not get the BCOO structure flags."""
        sparsity = SparsityPattern.from_coo([1, 0, 0], [1, 2, 0], (2, 3))
        bcoo = sparsity.to_bcoo()

        assert not bcoo.indices_sorted
        assert not bcoo.unique_indices

    def test_to_bcoo_no_structure_flags_for_duplicates(self):
        """User patterns with duplicate entries do not get the BCOO structure flags."""
        sparsity = SparsityPattern.from_coo([0, 0, 1], [2, 2, 0], (2, 3))
        bcoo = sparsity.to_bcoo()

        assert not bcoo.indices_sorted
        assert not bcoo.unique_indices


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

    def test_device_seeds_cached_per_dtype(self):
        """_device_seeds returns the same device array per dtype and matches _seed_matrix."""
        coloring = asdex.jacobian_coloring(lambda x: x * x, jnp.zeros(3))

        seeds = coloring._device_seeds(jnp.float32)
        assert seeds.dtype == jnp.float32
        assert seeds is coloring._device_seeds(jnp.float32)
        np.testing.assert_array_equal(np.asarray(seeds), coloring._seed_matrix)

        other = coloring._device_seeds(jnp.float16)
        assert other.dtype == jnp.float16
        assert other is not seeds

    def test_device_seeds_not_cached_under_jit(self):
        """A traced _device_seeds call must not poison the per-dtype cache.

        Under a jit trace, asarray returns a tracer;
        caching it would leak into later eager calls.
        """
        coloring = asdex.jacobian_coloring(lambda x: x * x, jnp.zeros(3))

        @jax.jit
        def traced(x):
            return coloring._device_seeds(jnp.float32) @ x

        traced(jnp.ones(3))
        seeds = coloring._device_seeds(jnp.float32)
        assert not isinstance(seeds, Tracer)


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
        assert (
            _render_dots(sparsity.m, sparsity.n, sparsity.rows, sparsity.cols)
            == "(empty)"
        )

    def test_render_dots_small_diagonal(self):
        """Dot rendering of small diagonal pattern."""
        sparsity = SparsityPattern.from_coo([0, 1, 2], [0, 1, 2], (3, 3))
        dots = _render_dots(sparsity.m, sparsity.n, sparsity.rows, sparsity.cols)

        # Should show diagonal pattern
        lines = dots.split("\n")
        assert len(lines) == 3
        assert "●" in lines[0]
        assert "⋅" in lines[0]

    def test_braille_empty_matrix(self):
        """Braille rendering of empty matrix."""
        sparsity = SparsityPattern.from_coo([], [], (0, 0))
        assert (
            _render_braille(sparsity.m, sparsity.n, sparsity.rows, sparsity.cols)
            == "(empty)"
        )

    def test_braille_large_matrix_downsamples(self):
        """Large matrices are downsampled in braille."""
        # Create 100x100 diagonal
        rows = list(range(100))
        cols = list(range(100))
        sparsity = SparsityPattern.from_coo(rows, cols, (100, 100))

        braille = _render_braille(
            sparsity.m,
            sparsity.n,
            sparsity.rows,
            sparsity.cols,
            max_height=10,
            max_width=20,
        )
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

        braille = _render_braille(
            sparsity.m,
            sparsity.n,
            sparsity.rows,
            sparsity.cols,
            max_height=20,
            max_width=40,
        )
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

    def test_repr_pytree_input_pattern(self):
        """__repr__ works for pattern from PyTree input."""

        def f(params):
            return params["a"] + params["b"] * 2

        params = {"a": np.zeros(2), "b": np.zeros(2)}
        sparsity = jacobian_sparsity(f, params)
        r = repr(sparsity)

        assert "SparsityPattern" in r
        assert "shape=(2, 4)" in r

    def test_str_pytree_input_pattern(self):
        """__str__ works for pattern from PyTree input."""

        def f(params):
            return params["a"] + params["b"] * 2

        params = {"a": np.zeros(2), "b": np.zeros(2)}
        sparsity = jacobian_sparsity(f, params)
        s = str(sparsity)

        assert "SparsityPattern" in s
        assert "●" in s or "⠀" in s  # Either dots or braille


class TestIntegration:
    """Integration tests with detection pipeline."""

    def test_jacobian_sparsity_returns_pattern(self):
        """jacobian_sparsity returns SparsityPattern."""

        def f(x):
            return jnp.array([x[0] * x[1], x[1] + x[2], x[2]])

        result = jacobian_sparsity(f, np.zeros(3))

        assert isinstance(result, SparsityPattern)
        assert result.shape == (3, 3)

        # Check sparsity pattern is correct
        expected = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
        np.testing.assert_array_equal(result.todense(), expected)

    def test_existing_tests_still_work(self):
        """Existing test patterns like .todense().astype(int) work."""

        def f(x):
            return x**2

        result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
        expected = np.eye(3, dtype=int)
        np.testing.assert_array_equal(result, expected)

    def test_print_sparsity(self):
        """Manual verification helper - prints sparsity pattern."""

        def f(x):
            return jnp.array([x[0] * x[1], x[1] + x[2], x[2]])

        sparsity = jacobian_sparsity(f, np.zeros(3))
        # This should print nicely with braille
        output = str(sparsity)
        assert len(output) > 0


# Save/Load tests


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


def test_save_load_hessian_coloring_decompression(tmp_path):
    """Hessian coloring survives save/load and produces correct results."""

    def f(x):
        return jnp.sum(x**2) + x[0] * x[1]

    x = jnp.array([1.0, 2.0, 3.0])
    coloring = asdex.hessian_coloring(f, x)
    expected = asdex.hessian_from_coloring(f, coloring, output_format="dense")(x)

    path = tmp_path / "hessian_coloring.npz"
    coloring.save(path)
    loaded = asdex.ColoredPattern.load(path)

    result = asdex.hessian_from_coloring(f, loaded, output_format="dense")(x)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_save_load_hessian_coloring_star_set_preserved(tmp_path):
    """star_set is preserved after save/load roundtrip."""

    def f(x):
        return jnp.sum(x**2) + x[0] * x[1]

    coloring = asdex.hessian_coloring(f, jnp.zeros(3))
    assert coloring.star_set is not None

    path = tmp_path / "hessian.npz"
    coloring.save(path)
    loaded = asdex.ColoredPattern.load(path)

    assert loaded.star_set is not None
    np.testing.assert_array_equal(loaded.star_set.star, coloring.star_set.star)
    np.testing.assert_array_equal(loaded.star_set.hub, coloring.star_set.hub)
    np.testing.assert_array_equal(loaded.star_set.edge_lo, coloring.star_set.edge_lo)
    np.testing.assert_array_equal(loaded.star_set.edge_hi, coloring.star_set.edge_hi)
    np.testing.assert_array_equal(loaded.star_set.edge_pos, coloring.star_set.edge_pos)


def test_load_legacy_symmetric_without_star_set_raises(tmp_path):
    """Loading old symmetric coloring without star/hub arrays raises ValueError."""
    sparsity = SparsityPattern.from_coo([0, 1, 0], [0, 1, 1], (2, 2))
    np.savez(
        tmp_path / "legacy.npz",
        rows=sparsity.rows,
        cols=sparsity.cols,
        shape=np.array(sparsity.shape),
        input_shape=np.array((2,)),
        colors=np.array([0, 1], dtype=np.int32),
        num_colors=np.array(2),
        symmetric=np.array(True),
        mode=np.array("fwd_over_rev"),
    )
    with pytest.raises(ValueError, match=r"star_set.*missing"):
        ColoredPattern.load(tmp_path / "legacy.npz")


def test_save_load_hessian_coloring_empty(tmp_path):
    """Hessian coloring with empty off-diagonal survives save/load."""

    def f(x):
        return jnp.sum(x**2)

    coloring = asdex.hessian_coloring(f, jnp.zeros(3))
    path = tmp_path / "empty_offdiag.npz"
    coloring.save(path)
    loaded = asdex.ColoredPattern.load(path)

    x = jnp.array([1.0, 2.0, 3.0])
    expected = asdex.hessian_from_coloring(f, coloring, output_format="dense")(x)
    result = asdex.hessian_from_coloring(f, loaded, output_format="dense")(x)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_save_load_hessian_coloring_all_modes(tmp_path, mode):
    """Hessian coloring survives save/load for all Hessian modes."""

    def f(x):
        return jnp.sum(x**2) + x[0] * x[1]

    x = jnp.array([1.0, 2.0, 3.0])
    coloring = asdex.hessian_coloring(f, x, mode=mode)
    expected = asdex.hessian_from_coloring(f, coloring, output_format="dense")(x)

    path = tmp_path / f"hessian_{mode}.npz"
    coloring.save(path)
    loaded = asdex.ColoredPattern.load(path)

    result = asdex.hessian_from_coloring(f, loaded, output_format="dense")(x)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_save_load_jacobian_coloring_no_star_set(tmp_path):
    """Jacobian coloring (non-symmetric) saves and loads without star_set."""

    def f(x):
        return x**2

    x = jnp.array([1.0, 2.0, 3.0])
    coloring = asdex.jacobian_coloring(f, x)
    assert coloring.star_set is None

    path = tmp_path / "jacobian.npz"
    coloring.save(path)
    loaded = asdex.ColoredPattern.load(path)

    assert loaded.star_set is None
    result = asdex.jacobian_from_coloring(f, loaded, output_format="dense")(x)
    expected = asdex.jacobian_from_coloring(f, coloring, output_format="dense")(x)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.parametrize("mode", ["fwd", "rev"], ids=["fwd", "rev"])
def test_save_load_jacobian_colored_roundtrip(tmp_path, mode):
    """Jacobian ColoredPattern survives a save/load roundtrip."""

    def f(x):
        return x**2

    original = asdex.jacobian_coloring(f, jnp.zeros(3), mode=mode)
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
    assert loaded.star_set is None


@pytest.mark.parametrize(
    "mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"], ids=str
)
def test_save_load_hessian_colored_roundtrip(tmp_path, mode):
    """Hessian ColoredPattern survives a save/load roundtrip with star_set."""

    def f(x):
        return jnp.sum(x**2)

    original = asdex.hessian_coloring(f, jnp.zeros(3), mode=mode)
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
    assert original.star_set is not None
    assert loaded.star_set is not None
    np.testing.assert_array_equal(loaded.star_set.star, original.star_set.star)
    np.testing.assert_array_equal(loaded.star_set.hub, original.star_set.hub)


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


# Multi-input pattern tests


def test_example_input_with_int_argnums():
    """example_input returns single aval when argnums is int."""

    def f(x, y):
        return x + y

    # argnums=0 (int) should return single aval, not tuple
    sparsity = asdex.jacobian_sparsity(f, np.zeros(3), np.zeros(3), argnums=0)
    example = sparsity.example_input
    # Should be a single ShapeDtypeStruct, not a tuple
    assert hasattr(example, "shape")
    assert example.shape == (3,)


def test_example_input_with_tuple_argnums():
    """example_input returns tuple of avals when argnums is tuple."""

    def f(x, y):
        return jnp.concatenate([x, y])

    # argnums=(0, 1) (tuple) should return tuple of avals
    sparsity = asdex.jacobian_sparsity(f, np.zeros(3), np.zeros(4), argnums=(0, 1))
    example = sparsity.example_input
    # Should be a tuple of ShapeDtypeStructs
    assert isinstance(example, tuple)
    assert len(example) == 2
    assert example[0].shape == (3,)
    assert example[1].shape == (4,)


def test_input_treedef_property():
    """input_treedef property returns the pytree structure of selected inputs."""

    def f(params):
        return params["a"] + params["b"]

    inputs = {"a": np.zeros(2), "b": np.zeros(2)}
    sparsity = asdex.jacobian_sparsity(f, inputs)
    treedef = sparsity.input_treedef
    # Should match the structure of dyn_avals
    assert treedef is not None


def test_save_load_multi_input_sparsity_roundtrip(tmp_path):
    """SparsityPattern with multi-input survives save/load."""

    def f(x, y):
        return x + y

    original = asdex.jacobian_sparsity(f, np.zeros(3), np.zeros(3), argnums=(0, 1))
    path = tmp_path / "multi.npz"
    original.save(path)

    loaded = asdex.SparsityPattern.load(path)

    np.testing.assert_array_equal(loaded.rows, original.rows)
    np.testing.assert_array_equal(loaded.cols, original.cols)
    assert loaded.shape == original.shape
    assert loaded.argnums == original.argnums
    assert loaded.leaf_shapes == original.leaf_shapes


def test_save_load_multi_input_coloring_roundtrip(tmp_path):
    """ColoredPattern with multi-input survives save/load and matches jax.jacobian."""

    def f(x, y):
        return x + y

    original = asdex.jacobian_coloring(f, np.zeros(3), np.zeros(3), argnums=(0, 1))
    path = tmp_path / "colored_multi.npz"
    original.save(path)

    loaded = asdex.ColoredPattern.load(path)

    x, y = np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])
    expected = jax.jacobian(f, argnums=(0, 1))(x, y)
    result = asdex.jacobian_from_coloring(f, loaded, output_format="dense")(x, y)
    assert _allclose_pytree(result, expected, rtol=1e-5)


def test_save_load_pytree_input_sparsity_roundtrip(tmp_path):
    """SparsityPattern with PyTree input survives save/load."""

    def f(d):
        return d["a"] + d["b"]

    original = asdex.jacobian_sparsity(f, {"a": np.zeros(2), "b": np.zeros(2)})
    path = tmp_path / "pytree_input.npz"
    original.save(path)

    loaded = asdex.SparsityPattern.load(path)

    np.testing.assert_array_equal(loaded.rows, original.rows)
    np.testing.assert_array_equal(loaded.cols, original.cols)
    assert loaded.shape == original.shape
    assert loaded.leaf_shapes == original.leaf_shapes


def test_save_load_pytree_input_coloring_roundtrip(tmp_path):
    """ColoredPattern with PyTree input survives save/load and matches jax.jacobian."""

    def f(d):
        return d["a"] + d["b"]

    original = asdex.jacobian_coloring(f, {"a": np.zeros(2), "b": np.zeros(2)})
    path = tmp_path / "colored_pytree_input.npz"
    original.save(path)

    loaded = asdex.ColoredPattern.load(path)

    d = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
    expected = jax.jacobian(f)(d)
    result = asdex.jacobian_from_coloring(f, loaded, output_format="dense")(d)
    assert _allclose_pytree(result, expected, rtol=1e-5)


def test_save_load_pytree_output_roundtrip(tmp_path):
    """SparsityPattern with PyTree output survives save/load and matches jax.jacobian."""

    def f(x):
        return {"a": x[:2], "b": x[2:]}

    original = asdex.jacobian_sparsity(f, np.zeros(4))
    path = tmp_path / "pytree_output.npz"
    original.save(path)

    loaded = asdex.SparsityPattern.load(path)

    x = np.arange(4.0)
    jax_jac = jax.jacobian(f)(x)
    jax_dense = np.abs(np.concatenate([jax_jac["a"], jax_jac["b"]], axis=0)) > 0
    np.testing.assert_array_equal(loaded.todense(), jax_dense.astype(np.int8))


def test_save_load_colored_pytree_output_roundtrip(tmp_path):
    """ColoredPattern with PyTree output survives save/load and matches jax.jacobian."""

    def f(x):
        return {"a": x[:2], "b": x[2:]}

    original = asdex.jacobian_coloring(f, np.zeros(4))
    path = tmp_path / "colored_pytree_output.npz"
    original.save(path)

    loaded = asdex.ColoredPattern.load(path)

    x = np.arange(4.0)
    expected = jax.jacobian(f)(x)
    result = asdex.jacobian_from_coloring(f, loaded, output_format="dense")(x)
    assert _allclose_pytree(result, expected, rtol=1e-5)


def test_save_load_multi_pytree_input_output(tmp_path):
    """save/load with 3 PyTree inputs and 2 PyTree outputs matches jax.jacobian."""

    def f(a, b, c):
        # a: {"x": (2,), "y": (3,)}
        # b: (4,)
        # c: {"z": (2,)}
        # output: (sum_xy, product_bc) as a tuple
        sum_xy = a["x"] + a["y"][:2]
        product_bc = b[:2] * c["z"]
        return (sum_xy, product_bc)

    a = {"x": np.zeros(2), "y": np.zeros(3)}
    b = np.zeros(4)
    c = {"z": np.zeros(2)}

    coloring = asdex.jacobian_coloring(f, a, b, c, argnums=(0, 1, 2))
    path = tmp_path / "multi_pytree.npz"
    coloring.save(path)

    loaded = asdex.ColoredPattern.load(path)

    a_val = {"x": np.array([1.0, 2.0]), "y": np.array([3.0, 4.0, 5.0])}
    b_val = np.array([6.0, 7.0, 8.0, 9.0])
    c_val = {"z": np.array([10.0, 11.0])}

    expected = jax.jacobian(f, argnums=(0, 1, 2))(a_val, b_val, c_val)
    result = asdex.jacobian_from_coloring(f, loaded, output_format="dense")(
        a_val, b_val, c_val
    )
    assert _allclose_pytree(result, expected, rtol=1e-5)


# Full API surface tests with multi-input patterns


def test_save_load_multi_input_jacobian_full_api(tmp_path):
    """Multi-input Jacobian: full API roundtrip via save/load."""

    def f(x, y):
        return jnp.concatenate([x * y[0], y * x.sum()])

    x, y = np.zeros(3), np.zeros(2)
    coloring = asdex.jacobian_coloring(f, x, y, argnums=(0, 1))
    path = tmp_path / "multi_jac.npz"
    coloring.save(path)
    loaded = asdex.ColoredPattern.load(path)

    x_val, y_val = np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])

    # jacobian_from_coloring
    result = asdex.jacobian_from_coloring(f, loaded, output_format="dense")(
        x_val, y_val
    )
    expected = jax.jacobian(f, argnums=(0, 1))(x_val, y_val)
    assert _allclose_pytree(result, expected, rtol=1e-5)

    # value_and_jacobian_from_coloring
    val, jac = asdex.value_and_jacobian_from_coloring(f, loaded, output_format="dense")(
        x_val, y_val
    )
    expected_val = f(x_val, y_val)
    np.testing.assert_allclose(val, expected_val, rtol=1e-5)
    assert _allclose_pytree(jac, expected, rtol=1e-5)

    # check_jacobian_correctness
    asdex.check_jacobian_correctness(f, (x_val, y_val), loaded)


def test_save_load_pytree_input_jacobian_full_api(tmp_path):
    """PyTree input Jacobian: full API roundtrip via save/load."""

    def f(d):
        return d["a"] * d["b"].sum() + d["b"][:2]

    d_shape = {"a": np.zeros(2), "b": np.zeros(3)}
    coloring = asdex.jacobian_coloring(f, d_shape)
    path = tmp_path / "pytree_jac.npz"
    coloring.save(path)
    loaded = asdex.ColoredPattern.load(path)

    d_val = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0, 5.0])}

    # jacobian_from_coloring
    result = asdex.jacobian_from_coloring(f, loaded, output_format="dense")(d_val)
    expected = jax.jacobian(f)(d_val)
    assert _allclose_pytree(result, expected, rtol=1e-5)

    # value_and_jacobian_from_coloring
    val, _jac = asdex.value_and_jacobian_from_coloring(
        f, loaded, output_format="dense"
    )(d_val)
    expected_val = f(d_val)
    np.testing.assert_allclose(val, expected_val, rtol=1e-5)

    # check_jacobian_correctness
    asdex.check_jacobian_correctness(f, d_val, loaded)


def test_save_load_multi_input_hessian_full_api(tmp_path):
    """Multi-input Hessian: full API roundtrip via save/load."""

    def f(x, y):
        return jnp.sum(x**2) + jnp.sum(y**2) + jnp.sum(x) * jnp.sum(y)

    x, y = np.zeros(3), np.zeros(2)
    coloring = asdex.hessian_coloring(f, x, y, argnums=(0, 1))
    path = tmp_path / "multi_hess.npz"
    coloring.save(path)
    loaded = asdex.ColoredPattern.load(path)

    x_val, y_val = np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])

    # hessian_from_coloring
    result = asdex.hessian_from_coloring(f, loaded, output_format="dense")(x_val, y_val)
    expected = jax.hessian(f, argnums=(0, 1))(x_val, y_val)
    assert _allclose_pytree(result, expected, rtol=1e-5)

    # value_and_hessian_from_coloring
    val, hess = asdex.value_and_hessian_from_coloring(f, loaded, output_format="dense")(
        x_val, y_val
    )
    expected_val = f(x_val, y_val)
    np.testing.assert_allclose(val, expected_val, rtol=1e-5)
    assert _allclose_pytree(hess, expected, rtol=1e-5)

    # check_hessian_correctness
    asdex.check_hessian_correctness(f, (x_val, y_val), loaded)


def test_save_load_pytree_input_hessian_full_api(tmp_path):
    """PyTree input Hessian: full API roundtrip via save/load."""

    def f(d):
        return jnp.sum(d["a"] ** 2) + jnp.sum(d["b"] ** 2) + jnp.dot(d["a"], d["b"][:2])

    d_shape = {"a": np.zeros(2), "b": np.zeros(3)}
    coloring = asdex.hessian_coloring(f, d_shape)
    path = tmp_path / "pytree_hess.npz"
    coloring.save(path)
    loaded = asdex.ColoredPattern.load(path)

    d_val = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0, 5.0])}

    # hessian_from_coloring
    result = asdex.hessian_from_coloring(f, loaded, output_format="dense")(d_val)
    expected = jax.hessian(f)(d_val)
    assert _allclose_pytree(result, expected, rtol=1e-5)

    # value_and_hessian_from_coloring
    val, hess = asdex.value_and_hessian_from_coloring(f, loaded, output_format="dense")(
        d_val
    )
    expected_val = f(d_val)
    np.testing.assert_allclose(val, expected_val, rtol=1e-5)
    assert _allclose_pytree(hess, expected, rtol=1e-5)

    # check_hessian_correctness
    asdex.check_hessian_correctness(f, d_val, loaded)


def test_save_load_preserves_argnums_int_vs_tuple(tmp_path):
    """Argnums type (int vs tuple) is preserved through save/load."""

    def f(x, y):
        return x + y

    x, y = np.zeros(3), np.zeros(3)

    # int argnums
    sp_int = asdex.jacobian_sparsity(f, x, y, argnums=0)
    path_int = tmp_path / "argnums_int.npz"
    sp_int.save(path_int)
    loaded_int = asdex.SparsityPattern.load(path_int)
    assert isinstance(loaded_int.argnums, int)
    assert loaded_int.argnums == 0

    # tuple argnums
    sp_tuple = asdex.jacobian_sparsity(f, x, y, argnums=(0, 1))
    path_tuple = tmp_path / "argnums_tuple.npz"
    sp_tuple.save(path_tuple)
    loaded_tuple = asdex.SparsityPattern.load(path_tuple)
    assert isinstance(loaded_tuple.argnums, tuple)
    assert loaded_tuple.argnums == (0, 1)


def test_save_load_nested_pytree_structure(tmp_path):
    """Nested PyTree structures survive save/load."""

    def f(d):
        return d["outer"]["inner"] + d["flat"]

    d_shape = {
        "outer": {"inner": np.zeros(2)},
        "flat": np.zeros(2),
    }
    coloring = asdex.jacobian_coloring(f, d_shape)
    path = tmp_path / "nested.npz"
    coloring.save(path)
    loaded = asdex.ColoredPattern.load(path)

    d_val = {
        "outer": {"inner": np.array([1.0, 2.0])},
        "flat": np.array([3.0, 4.0]),
    }

    result = asdex.jacobian_from_coloring(f, loaded, output_format="dense")(d_val)
    expected = jax.jacobian(f)(d_val)
    assert _allclose_pytree(result, expected, rtol=1e-5)


def test_save_load_mixed_dtypes(tmp_path):
    """Different dtypes in input_avals survive save/load."""

    def f(x):
        return x.astype(jnp.float32) * 2.0

    x_shape = np.zeros(3, dtype=np.float64)
    coloring = asdex.jacobian_coloring(f, x_shape)
    path = tmp_path / "dtypes.npz"
    coloring.save(path)
    loaded = asdex.ColoredPattern.load(path)

    assert loaded.sparsity.leaf_shapes == [(3,)]

    x_val = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    result = asdex.jacobian_from_coloring(f, loaded, output_format="dense")(x_val)
    expected = jax.jacobian(f)(x_val)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_save_load_multi_arg_correctness(tmp_path):
    """Multi-argument coloring produces correct Jacobian after save/load."""

    def f(x, y):
        return x * y + x**2

    x_shape, y_shape = np.zeros(3), np.zeros(3)
    coloring = asdex.jacobian_coloring(f, x_shape, y_shape, argnums=(0, 1))
    path = tmp_path / "multi_arg.npz"
    coloring.save(path)
    loaded = asdex.ColoredPattern.load(path)

    x_val, y_val = np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])
    result = asdex.jacobian_from_coloring(f, loaded, output_format="dense")(
        x_val, y_val
    )
    expected = jax.jacobian(f, argnums=(0, 1))(x_val, y_val)
    assert _allclose_pytree(result, expected, rtol=1e-5)


def test_list_container_preserved_after_load(tmp_path):
    """List containers are preserved after save/load."""

    def f(inputs):
        return inputs[0] + inputs[1]

    inputs_list = [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])]
    sparsity = jacobian_sparsity(f, inputs_list)

    path = tmp_path / "list_input.npz"
    sparsity.save(path)
    loaded = SparsityPattern.load(path)

    original_avals = sparsity.input_avals
    loaded_avals = loaded.input_avals

    assert isinstance(original_avals[0], list), "Original should have list container"
    assert isinstance(loaded_avals[0], list), "Loaded should preserve list container"


# Extraction index guards


def test_extraction_indices_neutral_color_raises():
    """A neutral (-1) color reaching extraction trips the host-side assertion.

    The decompression gather promises in-bounds indices,
    so a -1 color index would silently read garbage.
    Star-coloring postprocessing guarantees this cannot happen
    for colorings produced by the public API;
    the assertion guards hand-constructed or corrupted colorings.
    Extraction indices are computed eagerly,
    so the guard fires at construction time.
    """
    sparsity = SparsityPattern.from_coo([0, 1], [0, 1], (2, 2))
    with pytest.raises(AssertionError, match="neutral"):
        ColoredPattern(
            sparsity=sparsity,
            colors=np.array([0, -1], dtype=np.int32),
            num_colors=1,
            symmetric=False,
            mode="fwd",
        )


def test_hub_extraction_missing_edge_raises():
    """An off-diagonal pattern entry absent from the star set trips the guard.

    The batch edge lookup feeds a gather that promises in-bounds indices,
    so a missing edge must raise instead of silently reading garbage.
    Here the star set is empty,
    so the lookup position falls past the end of the edge arrays.
    Extraction indices are computed eagerly,
    so the guard fires at construction time.
    """
    sparsity = SparsityPattern.from_coo([0, 1], [1, 0], (2, 2))
    star_set = StarSet(
        star=np.empty(0, dtype=np.int32),
        hub=np.empty(0, dtype=np.int32),
    )
    with pytest.raises(AssertionError, match="missing"):
        ColoredPattern(
            sparsity=sparsity,
            colors=np.array([0, 1], dtype=np.int32),
            num_colors=2,
            symmetric=True,
            mode="fwd_over_rev",
            star_set=star_set,
        )


def test_hub_extraction_near_miss_edge_raises():
    """A near-miss edge lookup (in-bounds position, wrong key) trips the guard.

    The star set indexes edge ``(1, 2)`` while the pattern contains ``(0, 1)``,
    so the binary search lands on an in-bounds position with a mismatched key.
    Without the guard, the wrong hub would be picked
    and wrong values extracted silently.
    Extraction indices are computed eagerly,
    so the guard fires at construction time.
    """
    sparsity = SparsityPattern.from_coo([0, 1], [1, 0], (3, 3))
    star_set = StarSet(
        star=np.array([0], dtype=np.int32),
        hub=np.array([1], dtype=np.int32),
        edge_lo=np.array([1], dtype=np.int32),
        edge_hi=np.array([2], dtype=np.int32),
        edge_pos=np.array([0], dtype=np.int32),
    )
    with pytest.raises(AssertionError, match="missing"):
        ColoredPattern(
            sparsity=sparsity,
            colors=np.array([0, 1, 0], dtype=np.int32),
            num_colors=2,
            symmetric=True,
            mode="fwd_over_rev",
            star_set=star_set,
        )


# PyTree registration


def _banded_function(x):
    """Nonlinear function with a banded Jacobian, used by the pytree tests."""
    return x[:-1] * x[1:]


def test_sparsity_pattern_flatten_unflatten_roundtrip():
    """SparsityPattern roundtrips through tree_flatten and tree_unflatten.

    Uses a list container inside input_avals and tuple argnums
    to cover the aval flattening in the static aux data,
    since raw lists are not hashable and must travel flattened.
    """
    input_avals = (
        [ShapeDtypeStruct((2,), jnp.float32), ShapeDtypeStruct((1,), jnp.float32)],
        ShapeDtypeStruct((3,), jnp.float32),
    )
    original = SparsityPattern.from_coo(
        [0, 1, 2],
        [0, 1, 5],
        (3, 6),
        input_avals=input_avals,
        argnums=(0, 1),
    )

    leaves, treedef = jax.tree_util.tree_flatten(original)
    assert len(leaves) == 3  # rows, cols, _bcoo_indices

    roundtripped = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(roundtripped, SparsityPattern)
    np.testing.assert_array_equal(roundtripped.rows, original.rows)
    np.testing.assert_array_equal(roundtripped.cols, original.cols)
    np.testing.assert_array_equal(
        np.asarray(roundtripped._bcoo_indices), np.asarray(original._bcoo_indices)
    )
    assert roundtripped.shape == original.shape
    assert roundtripped.argnums == original.argnums
    assert roundtripped.input_avals == original.input_avals
    assert isinstance(roundtripped.input_avals[0], list)
    assert roundtripped._dyn_flat == original._dyn_flat
    assert roundtripped._entries_sorted_unique == original._entries_sorted_unique

    # Lazy caches are not part of the flat representation
    # and start out empty on the rebuilt instance.
    assert "col_to_rows" not in vars(roundtripped)
    assert "row_to_cols" not in vars(roundtripped)
    assert "_block_index_cache" not in vars(roundtripped)


def test_star_set_flatten_unflatten_roundtrip():
    """StarSet roundtrips through tree_flatten and tree_unflatten.

    The rebuilt instance must still answer edge and hub lookups.
    """
    original = StarSet(
        star=np.array([0], dtype=np.int32),
        hub=np.array([1], dtype=np.int32),
        edge_lo=np.array([0], dtype=np.int32),
        edge_hi=np.array([1], dtype=np.int32),
        edge_pos=np.array([0], dtype=np.int32),
    )

    leaves, treedef = jax.tree_util.tree_flatten(original)
    assert len(leaves) == 5

    roundtripped = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(roundtripped, StarSet)
    np.testing.assert_array_equal(roundtripped.star, original.star)
    np.testing.assert_array_equal(roundtripped.hub, original.hub)
    np.testing.assert_array_equal(roundtripped.edge_lo, original.edge_lo)
    np.testing.assert_array_equal(roundtripped.edge_hi, original.edge_hi)
    np.testing.assert_array_equal(roundtripped.edge_pos, original.edge_pos)
    assert roundtripped.edge_index(0, 1) == 0
    assert roundtripped.hub_vertex(0, 1) == 1


def test_colored_pattern_flatten_unflatten_roundtrip_nonsymmetric():
    """A Jacobian coloring roundtrips with all derived data reattached verbatim.

    Unflattening reattaches the flattened values instead of recomputing them,
    so the derived arrays on the rebuilt instance
    are the very same objects that were flattened.
    """
    x = jnp.arange(1.0, 6.0)
    original = asdex.jacobian_coloring(_banded_function, x)

    leaves, treedef = jax.tree_util.tree_flatten(original)
    roundtripped = jax.tree_util.tree_unflatten(treedef, leaves)

    assert isinstance(roundtripped, ColoredPattern)
    np.testing.assert_array_equal(roundtripped.colors, original.colors)
    assert roundtripped.num_colors == original.num_colors
    assert roundtripped.symmetric == original.symmetric
    assert roundtripped.mode == original.mode
    assert roundtripped.star_set is None

    assert roundtripped._gather_indices is original._gather_indices
    assert roundtripped._seed_matrix is original._seed_matrix

    # The per-dtype device seed memo is lazy and starts out empty.
    assert "_device_seed_cache" not in vars(roundtripped)

    # The rebuilt coloring computes the same Jacobian.
    jac_original = asdex.jacobian_from_coloring(
        _banded_function, original, output_format="dense"
    )(x)
    jac_roundtripped = asdex.jacobian_from_coloring(
        _banded_function, roundtripped, output_format="dense"
    )(x)
    np.testing.assert_array_equal(
        np.asarray(jac_roundtripped), np.asarray(jac_original)
    )


def test_colored_pattern_flatten_unflatten_roundtrip_symmetric():
    """A symmetric Hessian coloring roundtrips with its star set intact.

    An empty symmetric coloring carries star_set=None
    and therefore has a different treedef,
    which correctly reflects the structural difference.
    """

    def scalar_function(x):
        return x[0] * x[1] + x[2] ** 2 + x[3]

    x = jnp.arange(1.0, 5.0)
    original = asdex.hessian_coloring(scalar_function, x)
    assert original.symmetric
    assert original.star_set is not None

    leaves, treedef = jax.tree_util.tree_flatten(original)
    roundtripped = jax.tree_util.tree_unflatten(treedef, leaves)

    assert isinstance(roundtripped.star_set, StarSet)
    np.testing.assert_array_equal(roundtripped.star_set.star, original.star_set.star)
    np.testing.assert_array_equal(roundtripped.star_set.hub, original.star_set.hub)
    np.testing.assert_array_equal(roundtripped.colors, original.colors)

    hess_roundtripped = asdex.hessian_from_coloring(
        scalar_function, roundtripped, output_format="dense"
    )(x)
    np.testing.assert_allclose(
        np.asarray(hess_roundtripped),
        np.asarray(jax.hessian(scalar_function)(x)),
    )

    def linear_function(x):
        return x[0] + x[1] + x[2] + x[3]

    empty_symmetric = asdex.hessian_coloring(linear_function, x)
    assert empty_symmetric.symmetric
    assert empty_symmetric.star_set is None
    empty_treedef = jax.tree_util.tree_structure(empty_symmetric)
    assert empty_treedef != treedef

    empty_roundtripped = jax.tree_util.tree_unflatten(
        empty_treedef, jax.tree_util.tree_leaves(empty_symmetric)
    )
    assert empty_roundtripped.star_set is None
    assert empty_roundtripped.num_colors == 0


def test_colored_pattern_treedef_equality():
    """Structurally equal colorings share a treedef, different ones do not.

    Treedef equality is what makes jit caching work
    when a coloring is passed as an argument.
    """
    x = jnp.arange(1.0, 6.0)
    first = asdex.jacobian_coloring(_banded_function, x)
    second = asdex.jacobian_coloring(_banded_function, x)
    assert jax.tree_util.tree_structure(first) == jax.tree_util.tree_structure(second)

    wider = asdex.jacobian_coloring(_banded_function, jnp.arange(1.0, 7.0))
    assert jax.tree_util.tree_structure(first) != jax.tree_util.tree_structure(wider)

    reverse = asdex.jacobian_coloring(_banded_function, x, mode="rev")
    assert jax.tree_util.tree_structure(first) != jax.tree_util.tree_structure(reverse)


def test_colored_pattern_tree_map_identity():
    """An identity tree_map produces a fully working ColoredPattern."""
    x = jnp.arange(1.0, 6.0)
    original = asdex.jacobian_coloring(_banded_function, x)
    mapped = jax.tree_util.tree_map(lambda leaf: leaf, original)

    assert isinstance(mapped, ColoredPattern)
    jac_original = asdex.jacobian_from_coloring(
        _banded_function, original, output_format="dense"
    )(x)
    jac_mapped = asdex.jacobian_from_coloring(
        _banded_function, mapped, output_format="dense"
    )(x)
    np.testing.assert_array_equal(np.asarray(jac_mapped), np.asarray(jac_original))


def test_colored_pattern_through_jit():
    """A ColoredPattern passes through a jit boundary and stays usable.

    Unflattening inside the trace receives tracer leaves,
    so it must not run validation or numpy computations.
    The pattern returned from jit carries device arrays as leaves
    and must still decompress correctly.
    """
    x = jnp.arange(1.0, 6.0)
    coloring = asdex.jacobian_coloring(_banded_function, x)

    returned = jax.jit(lambda c: c)(coloring)
    assert isinstance(returned, ColoredPattern)
    assert returned.num_colors == coloring.num_colors
    assert returned.mode == coloring.mode
    np.testing.assert_array_equal(np.asarray(returned.colors), coloring.colors)

    jac_eager = asdex.jacobian_from_coloring(
        _banded_function, returned, output_format="dense"
    )(x)
    np.testing.assert_allclose(
        np.asarray(jac_eager), np.asarray(jax.jacobian(_banded_function)(x))
    )

    # decompress_data consumes the coloring inside jit with traced leaves.
    compressed = asdex.compressed_jacobian_from_coloring(_banded_function, coloring)(x)
    data_eager = asdex.decompress_data(compressed, coloring)
    data_jitted = jax.jit(asdex.decompress_data)(compressed, coloring)
    np.testing.assert_allclose(np.asarray(data_jitted), np.asarray(data_eager))


def test_colored_pattern_key_paths():
    """Flattening with paths yields attribute-named key entries."""
    x = jnp.arange(1.0, 6.0)
    coloring = asdex.jacobian_coloring(_banded_function, x)

    path_leaf_pairs, _ = jax.tree_util.tree_flatten_with_path(coloring)
    paths = [path for path, _ in path_leaf_pairs]
    rows_path = (
        jax.tree_util.GetAttrKey("sparsity"),
        jax.tree_util.GetAttrKey("rows"),
    )
    assert any(path == rows_path for path in paths)
    assert any(path == (jax.tree_util.GetAttrKey("colors"),) for path in paths)


def test_coloring_into_jitted_jacobian(output_format, to_dense):
    """A coloring built eagerly feeds a jitted sparse Jacobian computation.

    The coloring crosses the jit boundary as a pytree argument
    and the Jacobian computed inside jit matches
    both the eager sparse result and the JAX reference.
    """
    x = jnp.arange(1.0, 6.0)
    coloring = asdex.jacobian_coloring(_banded_function, x)

    @jax.jit
    def jitted_jacobian(coloring, x):
        return asdex.jacobian_from_coloring(
            _banded_function, coloring, output_format=output_format
        )(x)

    jac_jitted = jitted_jacobian(coloring, x)
    jac_eager = asdex.jacobian_from_coloring(
        _banded_function, coloring, output_format=output_format
    )(x)
    jac_reference = jax.jacobian(_banded_function)(x)

    np.testing.assert_allclose(to_dense(jac_jitted), to_dense(jac_eager))
    np.testing.assert_allclose(to_dense(jac_jitted), np.asarray(jac_reference))


def test_hessian_coloring_into_jitted_hessian(output_format, to_dense):
    """A symmetric Hessian coloring feeds a jitted sparse Hessian computation.

    The coloring crosses the jit boundary as a pytree argument,
    so the star-set arrays are traced leaves inside the trace,
    and the Hessian computed inside jit matches the JAX reference.
    """

    def scalar_function(x):
        return x[0] * x[1] + x[2] ** 2 + x[2] * x[3]

    x = jnp.arange(1.0, 5.0)
    coloring = asdex.hessian_coloring(scalar_function, x)
    assert coloring.symmetric
    assert coloring.star_set is not None

    @jax.jit
    def jitted_hessian(coloring, x):
        return asdex.hessian_from_coloring(
            scalar_function, coloring, output_format=output_format
        )(x)

    hess_jitted = jitted_hessian(coloring, x)
    hess_reference = jax.hessian(scalar_function)(x)
    np.testing.assert_allclose(to_dense(hess_jitted), np.asarray(hess_reference))


def test_pytree_input_coloring_into_jitted_jacobian(assert_trees_allclose):
    """A coloring for a pytree-input function crosses the jit boundary.

    The input structure travels in the static aux data,
    so leaf shapes and treedefs stay available while tracing
    with the index arrays as traced leaves.
    """

    def f(params):
        return params["w"] * params["b"][0]

    params = {"b": jnp.array([2.0]), "w": jnp.arange(1.0, 4.0)}
    coloring = asdex.jacobian_coloring(f, params)

    @jax.jit
    def jitted_jacobian(coloring, params):
        return asdex.jacobian_from_coloring(f, coloring, output_format="dense")(params)

    jac_jitted = jitted_jacobian(coloring, params)
    jac_reference = jax.jacobian(f)(params)
    assert_trees_allclose(jac_jitted, jac_reference)
