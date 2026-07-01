# decompression — Compress then Decompress

The final stage of the `detect -> color -> decompress` pipeline.
"Decompress" here is itself two sub-stages:
**compress** runs one VJP/JVP/HVP per color to produce the dense compressed matrix `B` of shape `(num_colors, dim)`,
then **decompress** scatters `B` back into the detected sparsity pattern.

This package owns the numerics of turning a `ColoredPattern` plus a function into a sparse Jacobian or Hessian,
and the public API surface that exposes it.

## Structure

- `__init__.py` — re-exports the public surface (see `_api.py`).
- `_api.py` — the user-facing entry points.
  Every function is a thin wrapper: normalize inputs, then delegate the numerics to the compress / decompress / evaluate stages.
- `_compress.py` — **stage 1**.
  Validate the call arguments and dtypes, short-circuit empty patterns,
  call the batched-AD engine in `_differentiation.py`, and return `B` (plus the forward value and aux).
  Also holds the input-prep helpers and the per-closure call cache shared with the composition layer.
- `_decompress.py` — **stage 2**, the pure consumer of `B`.
  Gathers `B` into the `(nnz,)` data vector in pattern order (`_decompress_data`),
  then either assembles the pytree/tensor output for the high-level functions (`_build_jacobian` / `_build_hessian`)
  or dispatches the flat `(m, n)` matrix for the public `decompress` (`_decompress_to_format`).
- `_evaluate.py` — composition of both stages for the one-shot `jacobian` / `hessian` / `value_and_*` family.
  One worker per direction (`_jacobian_with_value`, `_hessian_with_value`) always yields `(value, aux, matrix)`,
  and the four `_eval_*` entry points project that triple into the shape each caller expects.

The single shared layout fact, the second-axis length of `B`,
lives as the `ColoredPattern._compressed_dim` property in `src/asdex/_pattern.py`,
next to the other mode-derived facts (`_compresses_columns`, `_seed_matrix`).

The batched-AD engine itself lives outside this package, in `src/asdex/_differentiation.py`.
The compress stage calls into it; nothing else here touches raw AD.
The engine reads only the input structure and the seeds off the `ColoredPattern`,
never the nonzeros or the `OutputFormat`,
so it stays agnostic to how `B` is later decompressed.

## The core invariant: compress and decompress never import each other

`_compress.py` produces `B` and stops.
`_decompress.py` starts from `B` and never looks back at how it was produced:
it imports neither the compress side nor the AD engine.
Their only shared knowledge is the second-axis length of `B`,
which lives as `ColoredPattern._compressed_dim` in `src/asdex/_pattern.py`
so each side reaches it through the `ColoredPattern` it already holds, without importing the other.

`_evaluate.py` is the **only** module that depends on both stages.
It exists precisely so that gluing them together does not force compress and decompress to import each other and lose their independence.
Keep this boundary intact:
if you find yourself importing `_decompress` from `_compress` (or vice versa),
the shared fact belongs on `ColoredPattern` instead.

## The compressed matrix `B`

`B` has shape `(num_colors, dim)`,
where `dim` is the space that compression *preserves* — the opposite of the seeded space:

- `"fwd"` seeds the input space, so `B`'s columns are the output space of size `m`.
- `"rev"` and every Hessian mode seed the output / cotangent space,
  so `B`'s columns are the selected input space of size `n`.

`ColoredPattern._compressed_dim` is the single source of truth for this length.
Both sides consult it:
compress uses it to size the all-zero `_empty_compressed` on the empty-pattern short-circuit,
and decompress uses it to validate a caller-supplied `B` (`_validate_compressed`)
before the `PROMISE_IN_BOUNDS` gather that would otherwise read out of bounds rather than fail.

## Public API families

`_api.py` exposes four families, all built from the same two stages:

- **One-shot** — `jacobian`, `hessian`, `value_and_jacobian`, `value_and_hessian`:
  detect, color, then decompress in a single call.
- **From a coloring** — the `*_from_coloring` variants:
  skip detection and coloring and start from a pre-computed `ColoredPattern`.
- **Compressed** — `compressed_*` and `value_and_compressed_*` (plus their `*_from_coloring` variants):
  stop at `B` and hand it back as a plain `jax.Array`, so the caller can jit or post-process it.
  These take no `output_format`: formatting is `decompress`'s job.
- **Decompress consumers** — `decompress` and `decompress_data`:
  turn a caller-supplied `B` into a 2-D sparse matrix or the flat `(nnz,)` data vector in pattern order.

The one-shot, `*_from_coloring`, and decompress functions share their argument docs
through the `@_fill_doc` fragments in `src/asdex/_docstrings.py`.
The `compressed_*` functions deliberately cross-reference their non-compressed sibling for shared arguments instead,
so `B`'s layout is documented in exactly one place.

## The per-closure call cache

Each public entry point closes over one `_CallCache` dict (`_compress.py`) shared across calls of the returned function.
It memoizes work that depends only on the argument avals:
the `jax.eval_shape` output structure, the scalar-squeezing wrappers, and the internal jit core for host output formats.
The cache is **bypassed** (`None`) whenever call-time kwargs or non-traceable positional args were bound into `f`,
because those can change the output structure between calls with identical avals,
so nothing derived from `f` may be reused.

## Value, empty patterns, and float0

- The forward value rides the differentiation pass on the non-empty path, so it is nearly free.
  `need_value` only matters on the empty short-circuit,
  where a value-free caller skips the otherwise-wasted forward `f` call and receives `value=None`.
- Empty patterns (no output rows, or `nnz == 0`) short-circuit to an all-zero `B` / data vector,
  dtype-matched to the selected input leaves.
- Integer inputs under `allow_int=True` yield float0 cotangents,
  which cannot back a real array,
  so the decompress side falls back to a plain zero result of the default float dtype.

## Host output formats and the internal jit

numpy/scipy outputs cannot be wrapped in a caller-side `jax.jit`,
so `_evaluate.py` jits the array-valued core (compress + gather) internally instead (`_cached_jit_core`).
This is bypassed when jitting would be unsafe:
a fresh closure per call (`cache is None`),
or aux that may hold non-JAX types which cannot be jit outputs.

## Writing style

Use **semantic line breaks**: one sentence or clause per line, in docstrings, comments, and this file.
Focus comments on **why**, not what.
</content>
