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

- **The seam adds no flattening overhead.**
  The engine is already flat-in/flat-out today, so giving it its own module
  relocates code without adding work.
  `coloring._device_seeds(dtype)` is a flat `(num_colors, dim)` device array,
  and each vmapped kernel does only the pytree conversion JAX's AD forces at its
  boundary.
  In `"rev"` it unflattens the flat output-space seed into the `out_struct`
  cotangent, runs the VJP, then ravels the selected-input cotangents back to
  `(n_sel,)`.
  In `"fwd"` and the HVP modes it scatters the flat input-space seed into the
  input tangent pytree, runs the JVP/HVP, then ravels the output.
  Those `flatten_pytree`/`unflatten_to_pytree` calls (already in `_api_utils`)
  are inherent to JAX's pytree AD, not new cost, and `vmap` traces them once.
  All the engine needs to do this is a small *flatten spec*: `out_struct` plus
  the selected inputs' `leaf_shapes`, `leaf_sizes`, `input_avals`, and `argnums`.
  Every field is a pure function of `(input_avals, argnums)` (today they are
  `SparsityPattern` properties derived from exactly those), so they are
  AD-problem structure, not sparsity, and the engine rebuilds `B`'s flat layout
  without ever seeing the nonzeros.
  The compress layer computes the spec once per closure (cached), as today.

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
  (No extra flattening is incurred: see the flat-in/flat-out engine bullet above.)

- **The one-shot functions are built on the same core as the public stages.**
  `jacobian`/`hessian`/`value_and_*` (and their `*_from_coloring` variants) do not
  re-derive the numerics: they call the **same** compress core that backs
  `compressed_*` and the **same** gather primitive that backs `decompress_data`,
  so there is one implementation of compress and one of decompress.
  They do **not** literally call the public closures, for three deliberate
  reasons.
  The public `compressed_*` factory does its own input normalization and
  per-closure caching, so calling it from the one-shot would double both.
  `decompress` is flat `(m, n)` by the decision above, while the one-shot layers
  the pytree/tensor block-assembly tail (`_build_jacobian`/`_build_hessian`) on top
  of the shared gather, so for pytree outputs it cannot delegate to `decompress`.
  And for host output formats the shared compress and gather are fused under one
  `jax.jit` (`_cached_jit_core`, the perf path from #143), which wraps the core,
  not the public wrappers.
  The payoff is that the public API is the tested substrate: every existing e2e
  call already drives the shared core, so `compressed_*`/`decompress` inherit that
  coverage and the new tests only target the public-only deltas (validation,
  standalone use on a caller-supplied `B`, the full format set).

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

# Value-and-compressed — return (value, B) / ((value, aux), B); value rides the
# compression forward pass, so it is nearly free
def value_and_compressed_jacobian(f, *sample_args, argnums=0, has_aux=False,
                                  holomorphic=False, allow_int=False, mode=None,
                                  symmetric=False, chunk_size=None,
                                  **sample_kwargs): ...
def value_and_compressed_jacobian_from_coloring(f, coloring, *, has_aux=False,
                                                holomorphic=False,
                                                allow_int=False,
                                                chunk_size=None): ...
def value_and_compressed_hessian(f, *sample_args, argnums=0, has_aux=False,
                                 holomorphic=False, allow_int=False, mode=None,
                                 symmetric=True, chunk_size=None,
                                 **sample_kwargs): ...
def value_and_compressed_hessian_from_coloring(f, coloring, *, has_aux=False,
                                               holomorphic=False,
                                               allow_int=False,
                                               chunk_size=None): ...

# Decompression — gather compressed rows into sparse values, then format
def decompress_data(coloring, compressed): ...                  # -> (nnz,) jax.Array in sparsity order
def decompress(coloring, compressed, output_format="bcoo"): ... # -> 2-D matrix in any OutputFormat
```

**Avoiding docstring duplication.** This family shares many parameters
(`f`, `argnums`, `has_aux`, `holomorphic`, `allow_int`, `mode`, `symmetric`,
`chunk_size`), and the existing public API already repeats them.
Runtime `__doc__` composition (a decorator that stitches a shared `Args` block)
is **rejected**: `mkdocstrings` reads docstrings with griffe's *static* analyzer
(no dynamic option is set in `docs/mkdocs.yml`), so assembled docstrings would
not render, and runtime stitching also degrades `help()` and IDE hovers.
Instead keep **literal** docstrings but document each shared parameter **once**
on the canonical `jacobian`/`hessian`, and give the variants (`*_from_coloring`,
`compressed_*`, `value_and_*`) short docstrings that describe only what differs,
cross-referencing the canonical function through `autorefs`
(e.g. "See [`jacobian`][asdex.jacobian] for the shared arguments").
This is a pre-existing, API-wide concern, so apply the pattern to the new
functions and back-fill the existing family in the same docs pass.

- Each compression callable returns `B`, or `(B, aux)` when `has_aux=True`.
- `decompress_data` is the pure **gather primitive**: it returns a plain
  `jax.Array` of shape `(nnz,)` holding the sparse values in `coloring.sparsity`
  order, so `data[k]` is the entry at
  `(coloring.sparsity.rows[k], coloring.sparsity.cols[k])`.
  It exists **alongside** `decompress` because it is the jittable numeric core:
  it always returns a `jax.Array`, so it composes inside `jax.jit` and can feed a
  custom solver or sparse format, whereas `decompress` may return host objects
  (`numpy`/`scipy`) that cannot.
  It takes **no** `output_format` (the decompression-side analogue of
  `compressed_*` returning a raw `B`), and `decompress` is the thin host-format
  layer composed on top of it.
  Pair it with the already-public `coloring.sparsity.to_bcoo(data)` for a BCOO
  directly, or with `coloring.sparsity.rows`/`.cols` to assemble a custom format.
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

**Value-and-compressed variants (included).** The four `value_and_compressed_*`
functions return `(value, B)` / `((value, aux), B)`.
The value rides along the compression forward pass, so it is nearly free: the
Jacobian path already produces `y`, and the Hessian path reuses the existing
`_value_and_compute_hvps`, which gets value (and aux) for free in
`fwd_over_rev`/`rev_over_rev` and with one extra `f` call in `rev_over_fwd`.
They follow the shared-docstring rule above (short docstrings cross-referencing
`value_and_jacobian`/`value_and_hessian`).
A `compress(coloring, dense)` free function (dense → `B` via the seed matrix) is
deferred: it is only useful when one already has a dense matrix.

## Modularization refactor (behavior-preserving)

Split the conflated `decompression.py` into a sparsity-agnostic AD engine plus a
focused package. The decompress and composition halves are pure moves. The engine
extraction is the one real change: `_jacobian_rows`/`_jacobian_cols`/the HVP
kernels currently read `coloring._device_seeds(dtype)` and `coloring.sparsity`
internally, so they get parameterized on a seed matrix and `argnums` passed in by
the compress layer. Behavior-preserving, but signatures change. The public
`asdex.*` surface is unchanged except for the *added* symbols.
The split is done as a `git mv` to preserve blame (see "Git strategy" below).

The split has two parts. A top-level **`src/asdex/differentiation.py`** holds the
pure batched-AD engine. A **`src/asdex/decompression/`** package owns everything
sparsity-aware, following the same `__init__.py` + `_api.py` + private `_*.py`
layout as `coloring/` and `detection/`.

| Path | Owns | Public symbols |
|------|------|----------------|
| `src/asdex/differentiation.py` | Batched-AD engine: seed matrix in, derivative out | (none) |
| `decompression/_api.py` | Public surface only (thin wrappers + docstrings) | `jacobian`, `value_and_jacobian`, `hessian`, `value_and_hessian`, the `*_from_coloring` variants, `compressed_*`, `value_and_compressed_*`, `decompress`, `decompress_data` |
| `decompression/_compress.py` | Stage 1: coloring → seeds → drive the engine → `B`, plus shared input-prep helpers | (internal) |
| `decompression/_decompress.py` | Stage 3: gather + scatter/format over all `OutputFormat`s, pure consumer of `B` | (internal) |
| `decompression/_evaluate.py` | Composition of both stages: the four `_eval_*` plus per-closure caching | (internal) |
| `decompression/__init__.py` | Re-export the public surface (mirrors `coloring/__init__.py`) | (re-export) |

**Why `_evaluate.py` exists.** Keeping `_api.py` to just the public surface means
the `_eval_*` glue (validate, compress, decompress, build) has to live elsewhere.
It cannot go in `_compress.py` or `_decompress.py` without making those two import
each other, which would break their independence: compress produces `B`,
decompress consumes `B`, and neither needs the other. So it gets its own file. If
a flatter layout is later preferred, `_eval_*` can fold back into `_api.py` at the
cost of a heavier public file. The engine stays a top-level sibling for the same
reason in reverse: it is the one piece that knows nothing, so it sits at the
bottom as its own module rather than buried inside the package.

**Dependency graph (acyclic).** Edges point from importer to imported:

```
differentiation.py  ->  jax, _api_utils, modes        (leaf engine, no sparsity)
_compress.py        ->  differentiation, pattern, coloring, modes, _api_utils
_decompress.py      ->  pattern, modes, _api_utils     (independent of _compress)
_evaluate.py        ->  _compress, _decompress, pattern, modes
_api.py             ->  _evaluate, _compress, _decompress, detection, coloring
__init__.py         ->  _api
```

`_compress.py` and `_decompress.py` never import each other. `_api.py` is the only
file that reaches up to `detection` and `coloring` (for the one-shot
`jacobian`/`hessian`), exactly as `decompression.py` does today.

Move map:

- → `differentiation.py`: `_jacobian_rows` (becomes a batched VJP),
  `_jacobian_cols` (batched JVP), the per-mode HVP kernels and `_grad_with_*`
  pulled out of `_compute_hvps`/`_value_and_compute_hvps`, the seed/tangent/
  cotangent helpers (`_build_tangents_from_seed`, `_flatten_selected_cotangents`,
  `_flatten_grad_output`, `_build_grad_output_from_seed`), `_chunked_vmap`,
  `_output_dtype`. Selection is driven by a plain `argnums`, not a
  `SparsityPattern`, and the selected-input flatten order is taken from the single
  `_api_utils` convention so that `B`'s columns line up with `sparsity.cols`.
- → `_compress.py`: `_jacobian_compressed` (mode dispatch + seed building), the
  HVP mode dispatch and seed building from `_compute_hvps`/
  `_value_and_compute_hvps`, the compress-only evaluator behind `compressed_*`
  (validate + dtype checks + empty-pattern shortcut + compress core, stopping at
  `B`), and the shared input-prep helpers (`_validate_args`,
  `_cached_out_struct`/`_aval_key`, `_cached_scalar_fn`/`_scalar_with_aux`/
  `_cached_scalar_aux_fn`, `_selected_*`, `_uniform_selected_dtype`).
- → `_decompress.py`: `_decompress_data`, `_scatter_dense`, `_sparsity_to_scipy`,
  `_build_jacobian`/`_build_hessian`, `_assemble_*`, `_make_block_builder`,
  `_group_blocks_by_argnums`, `_is_simple_*`, the scipy assertions, and
  `_to_numpy_pytree`. The public `decompress_data` adds shape validation in front
  of the `_decompress_data` gather kernel, and `decompress` adds flat `(m, n)`
  format dispatch over the full `OutputFormat` set (`to_bcoo` / `_scatter_dense` /
  `_sparsity_to_scipy`, reusing `_assert_output_format` from `modes.py`).
  The internal `_eval_*` hot path keeps calling the unvalidated `_decompress_data`
  kernel directly (asdex produces `B` itself, so the boundary check is redundant
  there).
- → `_evaluate.py`: the four `_eval_*` functions, the per-closure caching and
  host-format jit-core (`_cached_jit_core`, `_build_*_core`, `_HOST_FORMATS`),
  and `_empty_data`.
- → `_api.py`: the public `jacobian`/`hessian`/`value_and_*` family and their
  `*_from_coloring` variants (each normalizes inputs, runs detection and coloring
  for the one-shot variants, and returns a thin closure delegating to
  `_evaluate`), the `compressed_*` and `value_and_compressed_*` functions
  (delegating to the `_compress.py` evaluator), and the public
  `decompress`/`decompress_data` (delegating to `_decompress.py`). Carries the
  Google-style docstrings (shared parameters documented once and cross-referenced
  per the docstring rule above).

**Git strategy (preserve blame, minimize diff).** Do the split as a `git mv`, not
a delete-and-recreate. `decompression.py` (a file) and `decompression/` (a
directory) are distinct names, so
`git mv src/asdex/decompression.py src/asdex/decompression/<target>.py` moves the
file into the new package in one step. Point the rename at whichever module
inherits the largest contiguous block (likely `_decompress.py` or `_evaluate.py`)
so `git blame` follows it, and confirm with `git diff -M --stat` that git reports
a rename rather than an add+delete. Commit the pure rename on its own, then
extract `differentiation.py`, `_compress.py`, the remaining package modules, and
the thin `__init__.py` in a follow-up commit. The extracted pieces lose per-line
blame continuity, which is unavoidable when one file becomes several, but the
rename keeps history on the bulk.

**Public surface placement.** `decompress`/`decompress_data` are thin enough that
their bodies could equally live in `_decompress.py` and be re-exported. We put the
definitions in `_api.py` so the whole public surface reads from one file, which
matches the user-facing intent of `_api.py`. The cost is one delegation hop for
those two functions.

Composition wins (less duplication):

- `compressed_jacobian_from_coloring` reuses the `_compress.py` core directly. Its
  evaluator is the existing `_eval_jacobian` **minus** the decompress-and-build
  tail (validation + dtype checks + empty-pattern → zeros + compress core).
- `_eval_jacobian` becomes "compute `B` via the compress core, then
  `_decompress_data` + `_build_jacobian`", so the two public stages and the
  high-level function share one compress core.
- The host-format jit-core hack stays in `_evaluate.py`; `compressed_*` need none
  of it (they return a plain `jax.Array`).

Suggested PR split: **PR1** the behavior-preserving refactor (extract the engine,
split `decompression.py` into the package, all existing tests green); **PR2** add
`compressed_*` + `decompress` + `decompress_data` + tests + docs. The maintainer
may combine them.

## Files to change

- **New:** `src/asdex/differentiation.py`, and the `src/asdex/decompression/`
  package (`__init__.py`, `_api.py`, `_compress.py`, `_decompress.py`,
  `_evaluate.py`).
- **Removed:** the flat `src/asdex/decompression.py` (becomes the
  `decompression/` package of the same import path).
- **Edit:** `src/asdex/__init__.py`. The existing `from asdex.decompression
  import (...)` keeps working through the package `__init__.py`; add
  `compressed_jacobian`, `compressed_jacobian_from_coloring`, `compressed_hessian`,
  `compressed_hessian_from_coloring`, the four `value_and_compressed_*` variants,
  `decompress`, and `decompress_data` to that import and to `__all__`.
- **Unchanged:** `src/asdex/pattern.py` (compress still reads
  `coloring._device_seeds`/`coloring.sparsity`), `src/asdex/coloring/`,
  `src/asdex/modes.py` (reuse `OutputFormat` + `_assert_output_format`),
  `src/asdex/_api_utils.py` (reuse `merge_sample_inputs`/`merge_args_kwargs`/
  `_ensure_index`/dtype validators and the selected-input flatten convention),
  `src/asdex/verify.py`.

## Edge cases

- **Empty pattern** (`num_colors == 0`, `nnz == 0`, or `m == 0`):
  compression returns an array of shape `(num_colors, dim)` (possibly with a
  zero axis) consistent with the non-empty path; `decompress` returns an empty
  sparse matrix.
  No separate `_empty.py` is warranted, and nothing empty-related goes in the
  sparsity-agnostic `differentiation.py`.
  The two empty shortcuts live on the side that owns each shape: the empty `B` is
  a one-liner `jnp.zeros((num_colors, dim), dtype)` in `_compress.py`, and the
  empty `(nnz,)` data vector stays in `_evaluate.py` as `_empty_data` (the
  empty-pattern branch of the `_eval_*` path), reusing the existing selected-dtype
  logic.
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

- **Round-trip and reference are covered by existing e2e (no new duplicates).**
  Because the one-shot `jacobian`/`hessian` are built on the same compress core
  and gather primitive that back the public stages (per the design decision and
  the Modularization section), every existing e2e call already drives that shared
  core across modes, `output_format`, `chunk_size`, and `has_aux`, and already
  runs `check_jacobian_correctness`/`check_hessian_correctness` on it, so
  `compressed_*`/`decompress` inherit the densified-matches-reference coverage.
  The existing tests stay as-is; we add no parallel round-trip/reference tests
  that would only re-cover them.
  The tests below instead target surface the e2e path does not reach.
- **Shape contract:** assert `B.shape == (coloring.num_colors, dim)` per the
  table above (rev/fwd/HVP), and `decompress_data(coloring, B).shape ==
  (coloring.sparsity.nnz,)`.
  Include pytree-input and pytree-output cases to confirm `B` stays a flat 2-D
  array regardless of input/output structure.
- **Gather primitive:** `decompress_data(c, B)` returns a jittable `(nnz,)`
  `jax.Array` whose dtype matches `B`. Feed it through the public
  `coloring.sparsity.to_bcoo` and confirm the densified result matches the
  reference, and that `jax.jit(decompress_data, ...)` compiles. Do **not** assert
  against `sparsity.rows`/`.cols` or rebuild the matrix by hand: that duplicates
  the internal scatter and leaks the implementation.
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

- **Reference:** add `::: asdex.compressed_jacobian` (+ `_from_coloring` and the
  `value_and_compressed_jacobian` pair) to `docs/reference/jacobian.md`, the
  hessian equivalents to `docs/reference/hessian.md`, and both
  `::: asdex.decompress` and `::: asdex.decompress_data` to a new "Decompression"
  section (in `reference/jacobian.md` or `reference/data-structures.md`). The
  `decompress` docstring enumerates the supported `OutputFormat`s; Google-style
  docstrings carry the detail, with shared parameters cross-referenced rather than
  repeated.
- **How-to:** one mirrored section, *"Skipping decompression"*, in
  both `docs/how-to/jacobians.md` and `docs/how-to/hessians.md`, with a runnable
  `exec="true"` snippet showing `compressed_*` → inspect `B` → `decompress`
  round-trip across a couple of output formats, plus `decompress_data` paired
  with `coloring.sparsity.rows`/`.cols` to build a custom COO triple.
  Semantic line breaks throughout. No em-dashes and no semicolons.

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
data = decompress_data(c, B)                            # (nnz,) jax.Array, jittable
# decompress_data is the jittable numeric core (stays inside jax.jit, feeds custom
# formats/solvers); decompress is the host-format layer (bcoo/numpy/scipy) on top.
J = decompress(c, B, output_format="bcoo")              # BCOO (m, n); any OutputFormat
ref = jacobian_from_coloring(f, c)(x).todense()
assert (c.sparsity.to_bcoo(data).todense() == ref).all()
assert (J.todense() == ref).all()
```
