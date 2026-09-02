# Test Suite

## Golden Rule

**Failing tests are gold.**
They reveal gaps in the implementation.
When a test fails, investigate and fix the code, not the test.
Never remove or simplify a test to make it pass.

## Structure

- Top-level test files (`test_*.py`) cover the public modules in `src/asdex/`.
  The `decompression/` package is split across two files by public-surface concern:
  `test_decompression.py` covers the sparse `jacobian` / `hessian` family (and their `*_from_coloring` variants and output formats) against JAX references,
  while `test_compression.py` covers the public `compressed_*` / `value_and_compressed_*` API returning a standalone `B`,
  plus `decompress` / `decompress_data` called directly on a caller-supplied `B`.
- `_interpret/` mirrors the handler modules: `_interpret/test_foo.py` tests `src/asdex/detection/_interpret/_foo.py`.
- `e2e/` contains end-to-end tests covering the full pipeline from public API through detection, coloring, and decompression.
- `smc/` cross-validates the coloring algorithms and their validators against
  [SparseMatrixColorings.jl](https://github.com/JuliaDiff/SparseMatrixColorings.jl) (SMC),
  the Julia package they were ported from.
- External-package handler tests live in subfolders (e.g., `_interpret/_equinox/`).

## Running Tests

Always run linting and type checking before tests:

```bash
uv run ruff check --fix .  # lint + auto-fix
uv run ruff format .       # format
uv run ty check            # type check
uv run pytest              # run tests (skips slow, benchmark, cutest, and smc by default: we only run these in CI)
```

## Markers

The custom markers and their descriptions live in a single source of truth:
the `markers` table under `[tool.pytest.ini_options]` in
[`pyproject.toml`](../pyproject.toml).
`--strict-markers` (also set there) rejects any unregistered marker,
so add a new one to that table before using it.

Use markers to run subsets of tests:

```bash
uv run pytest -m fallback        # Run only fallback tests
uv run pytest -m "not fallback"  # Skip fallback tests
uv run pytest -m "not slow"     # Skip slow tests
uv run pytest -m coloring        # Run only coloring tests
uv run pytest -m jacobian        # Run only sparse Jacobian tests
uv run pytest -m hessian         # Run only Hessian tests
```

## Test Utilities (conftest.py)

### Assertion helpers

- `assert_trees_allclose(actual, expected, rtol=1e-7, atol=0)`: Assert two pytrees have matching structure and allclose leaves. Automatically converts BCOO leaves to dense for comparison. Use as a fixture parameter in test functions.
- `to_dense(x)`: Convert a BCOO or scipy sparse array to dense, pass through other arrays. Use as a fixture parameter in test functions.

### Fixtures for parametrization

- `output_format`: Parametrizes over `"dense"` and `"bcoo"`.
- `all_output_format`: Parametrizes over all output formats, including `"numpy_dense"` and the scipy formats.
- `jacobian_mode`: Parametrizes over `"fwd"` and `"rev"`.
- `hessian_mode`: Parametrizes over `"fwd_over_rev"`, `"rev_over_fwd"`, and `"rev_over_rev"`.
- `chunk_size`: Parametrizes decompression chunking over `None`, `2`, and `3`.

Use these fixtures in test function signatures to automatically run tests across all variants.

### Sparsity helpers (`_utils.py`)

- `numerical_jacobian_sparsity(f, x, atol=1e-10, holomorphic=False)`: Reference sparsity pattern from `jax.jacobian`, thresholded by `atol`.
- `assert_jacobian_sparsity_exact(f, x, holomorphic=False)`: Assert the detected pattern equals the numerical pattern.
- `assert_jacobian_sparsity_conservative(f, x)`: Assert the detected pattern is a conservative superset of the numerical pattern.

## Conventions

- Each test function should have a docstring explaining what it tests.
- Tests documenting **expected future behavior** (TODOs) should use the `fallback` marker and include a `TODO(primitive)` comment explaining the precise expected behavior.
- **Whenever you discover a conservative pattern** (the handler produces a correct but overly dense result), you **must** document it with a `TODO(primitive)` comment showing the true precise pattern.
  Catching these is extremely valuable — each one is a concrete roadmap entry for improving sparsity detection.
- Tests documenting **known bugs** should use the `bug` marker and `pytest.raises` to assert the current (broken) behavior.
- When calling `assert_jacobian_sparsity_conservative`, also verify the detected pattern against a manually defined `expected` matrix using `np.testing.assert_array_equal`.
  This ensures both correctness (covers numerical) and precision (matches intent).

## Writing handler tests

Handler test files (`_interpret/test_*.py`) should cover:

- **Non-square shapes**: always use asymmetric shapes (e.g. `(3,4)` not `(4,4)`) so that dimension transposition bugs are caught.
- **Multiple dimensionalities**: 1D, 2D, 3D, 4D where applicable.
- **Broadcasting shapes**: size-1 dimensions that broadcast (e.g. `(3,4)` op `(3,1)`).
- **Degenerate shapes**: size-0 dimensions (zero-element arrays), size-1 dimensions, scalar inputs (where the primitive supports them).
  Size-0 tests verify the handler doesn't crash on empty arrays
  and returns an empty index set list.
- **Edge cases**: identity/trivial parameters, boundary parameter values.
- **Real-world usage patterns**: `jnp` functions that lower to the primitive under test.
- **Jacobian verification**: for at least one test per dimensionality, verify precision by comparing the detected pattern against `(np.abs(jax.jacobian(f)(x)) > 1e-10)` using `assert_array_equal`.
  Choose test functions that avoid local sparsity (e.g. multiply by zero) so the numerical Jacobian matches the structural pattern.
- **Inline matrix comments**: annotate expected sparsity matrices with inline comments
  explaining what each row computes. When consecutive rows share the same pattern
  (e.g. elementwise over a 2-element array), annotate only the first row of each group.
  ```python
  expected = np.array(
      [
          [1, 1, 1],  # carry_out = x[0] + x[1] + x[2]
          [0, 0, 0],  # ys[0] = carry_init = 0
          [1, 0, 0],  # ys[1] = x[0]
          [1, 1, 0],  # ys[2] = x[0] + x[1]
      ],
      dtype=int,
  )
  ```

## SparseMatrixColorings.jl Cross-Validation

asdex's greedy colorings are ports of SMC's.
Both use the `LargestFirst` vertex ordering with the same tie-breaking,
so on identical patterns they must produce *identical* colorings,
not merely colorings of the same quality.
`tests/smc/` asserts that equality on random and deterministic matrices,
and checks `check_coloring_cols` / `check_coloring_rows` / `check_coloring_symmetric`
against SMC's `structurally_orthogonal_columns` / `symmetrically_orthogonal_columns`.

```bash
uv sync --no-default-groups --group smc
uv run pytest tests/smc -m smc
```

SMC is called from Python through [PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl)'s
`juliacall` package.
Julia and SMC are installed on demand by `juliapkg` inside the session-scoped `smc` fixture,
which is the *only* place Julia is imported:
the `smc` marker is deselected by default and no module-level import touches Julia,
so the core suite never starts a Julia runtime.
CI runs this suite in its own `SMC` job.

Julia indexes colors from 1 and marks neutral vertices with 0,
while asdex indexes from 0 and marks neutral vertices with -1,
so colors crossing the bridge are shifted by one.

## CUTEst Integration Tests

[CUTEst](https://github.com/ralna/CUTEst) (Constrained and Unconstrained Testing Environment with safe threads)
is a standard benchmark suite for nonlinear optimization.
[`sif2jax`](https://github.com/johannahaffner/sif2jax) converts CUTEst SIF problem definitions into JAX-traceable functions.
These tests compare asdex's detected sparsity against CUTEst ground truth via `sif2jax`.

```bash
uv run pytest -m cutest  # requires sif2jax
```

Fixtures live in `tests/cutest_fixtures/` (`hessian/`, `jacobian_eq/`, `jacobian_ineq/`).
Regenerate with `tests/setup/generate_cutest_fixtures.py` (requires `pycutest`).

`EXPECTED_NNZ` tracks `(detected_nnz, target_nnz)` per problem.
Tests fail on both regressions and improvements to keep baselines current.
Update the tuple when a handler improvement reduces detected nnz.
