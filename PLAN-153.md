# Plan: Public compressed API (issue #153)

Tracking issue [#153](https://github.com/adrhill/asdex/issues/153)
(*feat: add public API to compute compressed Jacobians/Hessians*), milestone **JOSS**.

## Context

Automatic Sparse Differentiation runs in three stages:
**detect → color → decompress**, where the last stage is itself two steps:

1. **Compress** — evaluate one VJP/JVP/HVP per color, producing a dense
   *compressed matrix* `B` of shape `(num_colors, dim)`.
2. **Decompress** — scatter `B`'s entries back into the sparse pattern.

Users have asked to stop at step 1 and work with `B` directly
(custom sparse solvers, iterative schemes, debugging, cross-checking against
[SparseMatrixColorings.jl](https://github.com/gdalle/SparseMatrixColorings.jl)).
Today the compress/decompress split exists only as private helpers inside the
1985-line `src/asdex/decompression.py`, which bundles four concerns into one
file: the raw AD engine (batched VJP/JVP/HVP), the coloring-driven compression
that feeds it, the gather of `B` into `(nnz,)` data plus pytree/format assembly,
and the high-level composition with its per-closure caching.

The goal is **not** to bolt new surface onto a tangled file.
Tackled well, exposing the compress/decompress boundary as named public stages
*reduces* complexity: the existing one-shot functions become thin compositions
of two well-defined stages, the AD engine splits off into a sparsity-agnostic
`differentiation.py`, and the rest moves into a focused `decompression/` package.

## Design decisions

Driven by `CLAUDE.md` (minimize complexity, information hiding, pull complexity
downward, favor exceptions over wrong results):

- **The AD engine knows nothing about sparsity.**
  The batched VJP/JVP/HVP machinery moves to a top-level `differentiation.py`
  that takes a seed matrix and `argnums` and returns the batched derivative
  (plus the forward value and aux).
  It never imports `ColoredPattern`, `SparsityPattern`, or `OutputFormat`.
  Building seeds from a coloring and scattering `B` back both live in the
  `decompression/` package.
  This is the matrix-free-operator seam: the engine hides AD, chunking, and vmap,
  while the package hides how the coloring is exploited.
  The one invariant they share, that the selected-input flatten order matches
  `sparsity.cols`, is centralized in `_api_utils`.

- **`ColoredPattern` stays a pure data structure.**
  Issue #153 floats moving the scatter *onto* `ColoredPattern`; we deliberately
  do **not**.
  `decompress` and its gather primitive `decompress_data` are **free
  functions** in the `decompression/` package that *read* the pattern's cached
  index arrays (`coloring._gather_indices`, `coloring.sparsity`) without
  `pattern.py` importing any AD or output-format logic.
  This keeps the modules un-entangled.

- **Decompression operates on the flat 2-D matrix.**
  `decompress(coloring, B)` needs **only** `coloring + B` — no `out_struct`, no
  Jacobian-vs-Hessian branch, no pytree handling.
  The flat `(m, n)` sparse matrix always exists and is the natural domain of a
  "compressed matrix" (and matches the existing scipy 2-D-only constraint).
  Pytree/tensor-shaped block assembly stays a thin layer in the high-level
  `jacobian`/`hessian` functions, not in the public primitive.
  <!-- ISSUE: I'm not sure about this. What do the batched JVPs/VJPs look like on complex pytree inputs and outputs? We don't want to introduce any overhead by unnecessary flattening operations. -->

- **Compressed layout is `(num_colors, dim)` as-is.**
  Exactly what asdex computes; zero-copy, and `decompress` is a pure inverse with
  no orientation bookkeeping.
  Per mode:

  | API | mode | coloring | `B` shape | `dim` |
  |-----|------|----------|-----------|-------|
  | Jacobian | `"rev"` | rows | `(num_colors, n_sel)` | input size |
  | Jacobian | `"fwd"` | cols | `(num_colors, m)` | output size |
  | Hessian | any | cols | `(num_colors, n_sel)` | input size |

  where `n_sel` = Σ selected input leaf sizes, `m` = output size.
  *Trade-off to confirm:* SparseMatrixColorings.jl uses `(m, num_colors)` for
  column coloring and `(num_colors, n)` for row coloring (preserved dimension
  long, colors short).
  Matching it requires transposing the column-coloring cases on both compress
  output and decompress input — pure overhead and more code.
  We recommend `(num_colors, dim)`; flipping to the SMC.jl layout later is a
  localized change if cross-language consistency is judged more valuable.

## New public API

Naming mirrors the existing family (`jacobian` → `compressed_jacobian`).
Compression entry points return a raw `jax.Array` `B` (jit-able by the caller);
they take **no** `output_format` (formatting belongs to `decompress`).

```python
# Compression — returns B of shape (num_colors, dim)
def compressed_jacobian(f, *sample_args, argnums=0, has_aux=False,
                        holomorphic=False, allow_int=False, mode=None,
                        symmetric=False, chunk_size=None, **sample_kwargs): ...
def compressed_jacobian_from_coloring(f, coloring, *, has_aux=False,
                                      holomorphic=False, allow_int=False,
                                      chunk_size=None): ...
def compressed_hessian(f, *sample_args, argnums=0, has_aux=False,
                       holomorphic=False, allow_int=False, mode=None,
                       symmetric=True, chunk_size=None, **sample_kwargs): ...
def compressed_hessian_from_coloring(f, coloring, *, has_aux=False,
                                     holomorphic=False, allow_int=False,
                                     chunk_size=None): ...

# Decompression — gather compressed rows into sparse values, then format
def decompress_data(coloring, compressed): ...                  # -> (nnz,) jax.Array in sparsity order
def decompress(coloring, compressed, output_format="bcoo"): ... # -> 2-D matrix in any OutputFormat
```

- Each compression callable returns `B`, or `(B, aux)` when `has_aux=True`.
- `decompress_data` is the pure **gather primitive**: it returns a plain
  `jax.Array` of shape `(nnz,)` holding the sparse values in `coloring.sparsity`
  order, so `data[k]` is the entry at
  `(coloring.sparsity.rows[k], coloring.sparsity.cols[k])`.
  It is jittable and takes **no** `output_format` — the decompression-side
  analogue of `compressed_*` returning a raw `B`.
  Pair it with `coloring.sparsity.rows`/`.cols` to assemble any custom sparse
  format, or with the already-public `coloring.sparsity.to_bcoo(data)` for a
  BCOO directly.
- `decompress` is the **format-producing** entry point and **composes on the
  primitive** (`decompress = decompress_data` + format dispatch): it supports
  the **full `OutputFormat` set** already in `modes.py` — `"bcoo"` (default),
  `"dense"`, `"numpy_dense"`, `"scipy_coo"`, `"scipy_csr"`, `"scipy_csc"` — and
  returns the flat `(m, n)` matrix.
  Format dispatch lives in this one function (`to_bcoo` / `_scatter_dense` /
  `_sparsity_to_scipy`), so callers get every format from a single public call
  without touching the gather primitive.
- **Validation (favor exceptions over wrong results):** `decompress_data` checks
  `compressed.shape[0] == coloring.num_colors` and that axis 1 matches the
  mode's expected `dim`, raising `ValueError` before the `PROMISE_IN_BOUNDS`
  gather (a near-miss would otherwise read garbage); `decompress` inherits the
  check through it.

**Optional parity extension (maintainer's call):** `value_and_compressed_*`
(four functions returning `(value, B)` / `((value, aux), B)`).
The value rides free on the compression forward pass, so it is cheap, but it
doubles the compression surface — kept out of the core to minimize complexity.
A `compress(coloring, dense)` free function (dense → `B` via the seed matrix) is
likewise deferred; it is only useful when one already has a dense matrix.

## Modularization refactor (behavior-preserving)

Split the conflated `decompression.py` along its three real concerns.
This is **moves, not rewrites**; the public `asdex.*` surface is unchanged
except for the *added* symbols.

| New module | Owns | Public |
|------------|------|--------|
| `src/asdex/compression.py` | Stage 1: AD machinery computing `B` | `compressed_*` |
| `src/asdex/decompression.py` | Stage 3: gather (`decompress_data`) + scatter/format (`decompress`, all `OutputFormat`s) | `decompress`, `decompress_data` |
| `src/asdex/differentiation.py` | Composition of both stages | `jacobian`, `value_and_jacobian`, `hessian`, `value_and_hessian` + `*_from_coloring` |

Move map (functions already exist, just relocate):

- → `compression.py`: `_jacobian_compressed`, `_jacobian_rows`, `_jacobian_cols`,
  `_compute_hvps`, `_value_and_compute_hvps`, the seed/tangent/cotangent helpers
  (`_build_tangents_from_seed`, `_flatten_selected_cotangents`,
  `_flatten_grad_output`, `_build_grad_output_from_seed`, `_grad_with_*`),
  `_chunked_vmap`, dtype helpers (`_uniform_selected_dtype`, `_selected_*`).
- stays in `decompression.py`: `_decompress_data`, `_scatter_dense`,
  `_sparsity_to_scipy`, `_build_jacobian`/`_build_hessian`, `_assemble_*`,
  `_make_block_builder`, `_transpose_in_out_trees`, scipy assertions.
  Add the public `decompress_data` = shape validation + the `_decompress_data`
  gather kernel, and `decompress` = `decompress_data` + flat `(m, n)` format
  dispatch over the full `OutputFormat` set (`to_bcoo` / `_scatter_dense` /
  `_sparsity_to_scipy`, reusing `_assert_output_format` from `modes.py`).
  The internal `_eval_*` hot path keeps calling the unvalidated
  `_decompress_data` kernel directly (asdex produces `B` itself, so the boundary
  check is redundant there).
- → `differentiation.py`: the public `jacobian`/`hessian` family, the four
  `_eval_*` functions, and the per-closure caching helpers (`_cached_*`,
  `_aval_key`, `_build_*_core`, `_HOST_FORMATS`).

Composition wins (less duplication):

- `compressed_jacobian_from_coloring` reuses `_jacobian_compressed` directly;
  its evaluator is the existing `_eval_jacobian` **minus** the
  decompress-and-build tail (validation + dtype checks + empty-pattern → zeros
  + compress core).
- `_eval_jacobian` becomes "compute `B` via the compressed evaluator, then
  `_decompress_data` + `_build_jacobian`", so the two public stages and the
  high-level function share one compression core.
- The internal jit-core hack for host formats stays in `differentiation.py`;
  `compressed_*` need none of it (they return a plain `jax.Array`).

Suggested PR split: **PR1** pure refactor (split modules, expose private cores,
all existing tests green); **PR2** add `compressed_*` + `decompress` + tests +
docs. The maintainer may combine them.

## Files to change

- **New:** `src/asdex/compression.py`, `src/asdex/differentiation.py`.
- **Edit:** `src/asdex/decompression.py` (slim to the decompress stage, add
  `decompress` and `decompress_data`), `src/asdex/__init__.py` (import the new
  symbols from their new homes; add `decompress` and `decompress_data` to
  `__all__`).
- **Unchanged:** `src/asdex/pattern.py`, `src/asdex/coloring/`, `src/asdex/modes.py`
  (reuse `OutputFormat` + `_assert_output_format`), `src/asdex/_api_utils.py`
  (reuse `merge_sample_inputs`/`merge_args_kwargs`/`_ensure_index`/dtype
  validators), `src/asdex/verify.py`.

## Edge cases

- **Empty pattern** (`num_colors == 0`, `nnz == 0`, or `m == 0`):
  compression returns an array of shape `(num_colors, dim)` (possibly with a
  zero axis) consistent with the non-empty path; `decompress` returns an empty
  sparse matrix. Mirror the existing `_empty_data` dtype logic.
- **float0 cotangents** (`allow_int=True`): already handled in `_scatter_dense`
  and `_build_*`; `decompress` inherits this.
  `decompress_data` returns the raw gathered values (float0 preserved), matching
  the compressed array's dtype.
- **Symmetric (star) colorings:** `_gather_indices` already encodes hub-based
  extraction; `decompress` works unchanged (uses `unique_indices=not symmetric`).

## Tests

Add `tests/test_compression.py` (mirrors `tests/test_decompression.py`
conventions) and extend `tests/e2e/`. Reuse the `conftest.py` fixtures
`jacobian_mode`, `hessian_mode`, `chunk_size`, `all_output_format`, `to_dense`,
`assert_trees_allclose`.

- **Round-trip (core property):** for flat 2-D `f`,
  `decompress(coloring, compressed_jacobian_from_coloring(f, coloring)(x))`
  equals `jacobian_from_coloring(f, coloring)(x)` across all `output_format`,
  modes, `chunk_size`, and `has_aux`. Same for Hessian.
- **Reference check:** `decompress(...).todense()` matches `jax.jacobian` /
  `jax.hessian`; also drive `check_jacobian_correctness` /
  `check_hessian_correctness` on the composed path to confirm the refactor.
- **Shape contract:** assert `B.shape == (coloring.num_colors, dim)` per the
  table above (rev/fwd/HVP), and `decompress_data(coloring, B).shape ==
  (coloring.sparsity.nnz,)`.
- **Gather primitive:** `coloring.sparsity.to_bcoo(decompress_data(c, B))`
  equals `decompress(c, B, "bcoo")`, and the COO triple
  `(sparsity.rows, sparsity.cols, decompress_data(c, B))` reconstructs the
  reference matrix; values come back in `sparsity` order.
- **Output formats:** `decompress` round-trips through **every** `OutputFormat`
  (`"bcoo"`, `"dense"`, `"numpy_dense"`, `"scipy_coo"`, `"scipy_csr"`,
  `"scipy_csc"`) — reuse the `all_output_format` fixture — each matching the
  reference once densified.
- **Validation:** wrong `num_colors`/`dim` raise from `decompress_data` (and
  therefore from `decompress`); unknown `output_format` and scipy-without-scipy
  raise from `decompress` (existing `ImportError` pattern).
- **Edge cases:** empty/zero patterns, `has_aux`, pytree inputs (compression
  still returns flat `B`; `decompress` still returns flat `(m, n)`).

## Docs

- **Reference:** add `::: asdex.compressed_jacobian` (+ `_from_coloring`) to
  `docs/reference/jacobian.md`, the hessian equivalents to
  `docs/reference/hessian.md`, and both `::: asdex.decompress` and
  `::: asdex.decompress_data` to a new "Decompression" section (in
  `reference/jacobian.md` or `reference/data-structures.md`). The `decompress`
  docstring enumerates the supported `OutputFormat`s; Google-style docstrings
  carry the detail.
- **How-to:** one mirrored section, *"Working with the Compressed Matrix"*, in
  both `docs/how-to/jacobians.md` and `docs/how-to/hessians.md`, with a runnable
  `exec="true"` snippet showing `compressed_*` → inspect `B` → `decompress`
  round-trip across a couple of output formats, plus `decompress_data` paired
  with `coloring.sparsity.rows`/`.cols` to build a custom COO triple.
  Semantic line breaks throughout.
- **Explanation:** add a short pointer from `docs/explanation/asd.md` (or
  `coloring.md`) defining `B = A·S` / `Sᵀ·A` and the `(num_colors, dim)` layout.
- Coordinate with PLAN-152 (docs overhaul) to avoid conflicts; #153 is a feature
  and is explicitly out of scope there.

## Verification

```bash
uv run ruff check --fix . && uv run ruff format .   # lint first (firm convention)
uv run ty check
uv run pytest                                        # full suite, incl. new round-trip tests
uv run mkdocs build --strict -f docs/mkdocs.yml      # exec snippets + link/anchor check
```

Manual smoke test:

```python
import jax.numpy as jnp
from asdex import jacobian_coloring, compressed_jacobian_from_coloring, decompress, decompress_data, jacobian_from_coloring

f = lambda x: (x[1:] - x[:-1]) ** 2
x = jnp.arange(1.0, 11.0)
c = jacobian_coloring(f, x)
B = compressed_jacobian_from_coloring(f, c)(x)          # (num_colors, dim)
data = decompress_data(c, B)                            # (nnz,) in sparsity order
J = decompress(c, B, output_format="bcoo")              # BCOO (m, n); any OutputFormat
ref = jacobian_from_coloring(f, c)(x).todense()
assert (c.sparsity.to_bcoo(data).todense() == ref).all()
assert (J.todense() == ref).all()
```
