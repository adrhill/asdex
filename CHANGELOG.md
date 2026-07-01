# asdex

## Version `v0.5.0`
This is not a breaking release for most users and has been tagged conservatively only for those importing from internal file paths.

* ![Feature][badge-feature] Add compressed-differentiation API (`compressed_jacobian`, `compressed_hessian`, their `value_and_*` and `*_from_coloring` variants) returning the raw compressed matrix `B` ([#155])
* ![Feature][badge-feature] Add `decompress` and `decompress_data` to turn a compressed matrix back into a sparse matrix or its raw values respectively ([#155])
* ![Bugfix][badge-bugfix] Compute the primal value for free in reverse-over-forward Hessians, so `value_and_hessian(mode="rev_over_fwd")` no longer costs an extra function evaluation ([#155])
* ![Bugfix][badge-bugfix] Correct sparse verification for non-leading tuple `argnums` ([#158])
* ![Documentation][badge-docs] Update README and how-to guides with new `v0.3` and `v0.4` features ([#154])
* ![Maintenance][badge-maintenance] Make internal modules private.
  The top-level `asdex` namespace re-exports every public symbol unchanged, so this only affects code importing from internal file paths ([#158]).

```python
coloring = asdex.jacobian_coloring(f, x)
B = asdex.compressed_jacobian_from_coloring(f, coloring)(x)  # compressed matrix
J = asdex.decompress(B, coloring)                            # back to a sparse matrix
data = asdex.decompress_data(B, coloring)                    # raw values
```

## Version `v0.4.0`
* ![BREAKING][badge-breaking] `StarSet` now stores edge keys as arrays; `reconstruct_edge_index` is renamed to `reconstruct_edge_arrays` and `edge_index` becomes a lookup method ([#143])
* ![Feature][badge-feature] Add `numpy_dense`, `scipy_coo`, `scipy_csr`, and `scipy_csc` output formats, with scipy as an optional dependency ([#142])
* ![Feature][badge-feature] Add handler for the `remat2` primitive (`jax.checkpoint` / `jax.remat`) ([#141])
* ![Enhancement][badge-enhancement] Reduce per-call decompression overhead and speed up symmetric coloring ([#143])
* ![Enhancement][badge-enhancement] JIT-compile the numpy and scipy output formats internally, since they cannot be wrapped in `jax.jit` by the caller ([#143])
* ![Bugfix][badge-bugfix] Raise clear `TypeError`s for integer and `allow_int` Hessian inputs and for mixed input dtypes in forward and Hessian modes ([#143])

## Version `v0.3.3`
* ![Feature][badge-feature] Add handlers for bitwise, random, cumulative, and linalg primitives ([#140])

## Version `v0.3.2`
* ![Feature][badge-feature] Add `chunk_size` parameter for bounded memory usage in Jacobian and Hessian computation ([#139])

Example:
```python
# Limit parallelism to 128 colors at a time (reduces peak memory)
J = asdex.jacobian(f, x, chunk_size=128)(x)
```

## Version `v0.3.1`
* ![Feature][badge-feature] Add precise handlers for `stack` and `unstack` primitives ([#137])
* ![Feature][badge-feature] Add handlers for scalar special functions (`erfc`, `erf_inv`, `digamma`, `lgamma`, `bessel_i0e`, `bessel_i1e`, `polygamma`) ([#127])
* ![Bugfix][badge-bugfix] Support keyword arguments at detection and call time ([#134])
* ![Bugfix][badge-bugfix] Reject `allow_int=True` in forward mode ([#133])
* ![Bugfix][badge-bugfix] Handle data-dependent indices in `gather` ([#132])
* ![Bugfix][badge-bugfix] Handle empty `axis=()` as identity in `reduce` ([#130])
* ![Bugfix][badge-bugfix] Handle `clamp` primitive with non-zero derivative ([#129])
* ![Maintenance][badge-maintenance] Add parametrized tests for elementwise handlers ([#128])
* ![Documentation][badge-docs] Add Zenodo citation request and funding acknowledgment ([#135])

## Version `v0.3.0`
* ![BREAKING][badge-breaking] API now requires sample inputs instead of `input_shape` parameter ([#105])
* ![Feature][badge-feature] Support pytree / multi-positional-input functions via `argnums` ([#105])
* ![Feature][badge-feature] Add `has_aux`, `holomorphic`, `allow_int` kwargs matching JAX semantics ([#105])
* ![Feature][badge-feature] Support PyTree outputs in Jacobian/Hessian computation ([#105])
* ![Enhancement][badge-enhancement] Support scalar inputs (0-dimensional arrays) ([#105])

```python
# Before (v0.2)
J = asdex.jacobian(f, input_shape=x.shape)(x)

# After (v0.3) — sample inputs like jax.jacobian
J = asdex.jacobian(f, x)(x)
J = asdex.jacobian(f, x, y, argnums=(0, 1))(x, y)
```

## Version `v0.2.0`
* ![BREAKING][badge-breaking] `color_symmetric` now returns `(colors, num_colors, star_set)` instead of `(colors, num_colors)` ([#104])
* ![Feature][badge-feature] Promote `check_coloring_rows`, `check_coloring_cols`, `check_coloring_symmetric` to public API in `verify.py` ([#104])
* ![Feature][badge-feature] Add `postprocess=True` kwarg to `color_symmetric`, demoting unused hub colors to reduce HVP count ([#104])
* ![Enhancement][badge-enhancement] `ColoredPattern` gains optional `star_set` field for O(1) hub lookup during Hessian decompression ([#104])
* ![Enhancement][badge-enhancement] 7–9× coloring speedup via numba JIT (`@njit(cache=True)`) ([#104])
* ![Bugfix][badge-bugfix] Fix star constraint for internal vertices in `color_symmetric` ([#102])
* ![Maintenance][badge-maintenance] Move coloring validators from tests to library and tidy test helpers ([#103])

## Version `v0.1.8`
* ![Feature][badge-feature] Add `output_format` kwarg for optional dense decompression ([#100])
* ![Enhancement][badge-enhancement] Smaller decompression jaxpr by using `lax.gather` ([#99])
* ![Documentation][badge-docs] Use `jax.jit` in all examples ([#101])

## Version `v0.1.7`
* ![Feature][badge-feature] Add `spy` matplotlib visualization ([#97])
* ![Documentation][badge-docs] Add coloring algorithm references and subsections to API docs ([#94])
* ![Documentation][badge-docs] Fix incorrect API references in amortization section ([#96])
* ![Maintenance][badge-maintenance] Upgrade `ty` and pin linter versions ([#95])

## Version `v0.1.6`
* ![Feature][badge-feature] Export `color_rows`, `color_cols`, and `color_symmetric` in public API ([669b5c])
* ![Feature][badge-feature] Validate and coerce inputs in `*_coloring_from_sparsity` ([#90])
* ![Bugfix][badge-bugfix] Preserve aspect ratio in braille sparsity display ([#91])

## Version `v0.1.5`
* ![Feature][badge-feature] Add `value_and_jacobian` / `value_and_hessian` API ([#89])
* ![Enhancement][badge-enhancement] Avoid redundant forward pass in `jacobian` and `hessian` ([#89])

## Version `v0.1.4`
* ![Bugfix][badge-bugfix] Handle zero-sized arrays in `broadcast_in_dim`, `sort`, and `gather` ([#88])

## Version `v0.1.3`
* ![Feature][badge-feature] Per-timestep forward simulation in `scan` handler ([#87])
* ![Feature][badge-feature] Per-element branch selection in `select_n` for const predicates ([#85])
* ![Feature][badge-feature] Propagate const values through `squeeze` ([#84])
* ![Feature][badge-feature] Merge value bounds in `select_n` for dynamic predicates ([#83])

## Version `v0.1.2`
* ![Feature][badge-feature] Handle all `ScatterDimensionNumbers` configurations ([#81])
* ![Feature][badge-feature] Zero-skipping and bounds propagation for `div`, `mul`, `integer_pow` ([#80])
* ![Feature][badge-feature] Track value bounds for dynamic-index sparsity ([#78])
* ![Bugfix][badge-bugfix] Handle all `GatherDimensionNumbers` configurations ([#77])
* ![Maintenance][badge-maintenance] Rename interpreter state types and variables ([#79])

## Version `v0.1.1`
* ![Feature][badge-feature] Add `cumsum` primitive handler ([#76])
* ![Feature][badge-feature] Add `erf` to unary elementwise dispatch ([#75])
* ![Bugfix][badge-bugfix] Fix `dot_general` handler for scalar operands ([#75])
* ![Bugfix][badge-bugfix] Fix `gather` handler for wrong ndim in single-dim path ([#75])
* ![Bugfix][badge-bugfix] Handle `batch_group_count > 1` in conv handler ([#73])
* ![Enhancement][badge-enhancement] Factor out primal computation in `fwd_over_rev` and `rev_over_rev` ([#72])
* ![Maintenance][badge-maintenance] Suppress expected warnings for clean pytest output ([#74])
* ![Documentation][badge-docs] Update for PyPI release ([#71])

## Version `v0.1.0`
* ![Feature][badge-feature] Initial release ([#70])


[#158]: https://github.com/adrhill/asdex/pull/158
[#155]: https://github.com/adrhill/asdex/pull/155
[#154]: https://github.com/adrhill/asdex/pull/154
[#143]: https://github.com/adrhill/asdex/pull/143
[#142]: https://github.com/adrhill/asdex/pull/142
[#141]: https://github.com/adrhill/asdex/pull/141
[#140]: https://github.com/adrhill/asdex/pull/140
[#139]: https://github.com/adrhill/asdex/pull/139
[#137]: https://github.com/adrhill/asdex/pull/137
[#135]: https://github.com/adrhill/asdex/pull/135
[#134]: https://github.com/adrhill/asdex/pull/134
[#133]: https://github.com/adrhill/asdex/pull/133
[#132]: https://github.com/adrhill/asdex/pull/132
[#130]: https://github.com/adrhill/asdex/pull/130
[#129]: https://github.com/adrhill/asdex/pull/129
[#128]: https://github.com/adrhill/asdex/pull/128
[#127]: https://github.com/adrhill/asdex/pull/127
[#105]: https://github.com/adrhill/asdex/pull/105
[#104]: https://github.com/adrhill/asdex/pull/104
[#103]: https://github.com/adrhill/asdex/pull/103
[#102]: https://github.com/adrhill/asdex/pull/102
[#101]: https://github.com/adrhill/asdex/pull/101
[#100]: https://github.com/adrhill/asdex/pull/100
[#99]: https://github.com/adrhill/asdex/pull/99
[#97]: https://github.com/adrhill/asdex/pull/97
[#96]: https://github.com/adrhill/asdex/pull/96
[#95]: https://github.com/adrhill/asdex/pull/95
[#94]: https://github.com/adrhill/asdex/pull/94
[#91]: https://github.com/adrhill/asdex/pull/91
[#90]: https://github.com/adrhill/asdex/pull/90
[#89]: https://github.com/adrhill/asdex/pull/89
[#88]: https://github.com/adrhill/asdex/pull/88
[#87]: https://github.com/adrhill/asdex/pull/87
[#85]: https://github.com/adrhill/asdex/pull/85
[#84]: https://github.com/adrhill/asdex/pull/84
[#83]: https://github.com/adrhill/asdex/pull/83
[#81]: https://github.com/adrhill/asdex/pull/81
[#80]: https://github.com/adrhill/asdex/pull/80
[#79]: https://github.com/adrhill/asdex/pull/79
[#78]: https://github.com/adrhill/asdex/pull/78
[#77]: https://github.com/adrhill/asdex/pull/77
[#76]: https://github.com/adrhill/asdex/pull/76
[#75]: https://github.com/adrhill/asdex/pull/75
[#74]: https://github.com/adrhill/asdex/pull/74
[#73]: https://github.com/adrhill/asdex/pull/73
[#72]: https://github.com/adrhill/asdex/pull/72
[#71]: https://github.com/adrhill/asdex/pull/71
[#70]: https://github.com/adrhill/asdex/pull/70
[669b5c]: https://github.com/adrhill/asdex/commit/669b5c

<!--
# Badges
![BREAKING][badge-breaking]
![Deprecation][badge-deprecation]
![Feature][badge-feature]
![Enhancement][badge-enhancement]
![Bugfix][badge-bugfix]
![Experimental][badge-experimental]
![Maintenance][badge-maintenance]
![Documentation][badge-docs]
-->

[badge-breaking]: https://img.shields.io/badge/BREAKING-red.svg
[badge-deprecation]: https://img.shields.io/badge/deprecation-orange.svg
[badge-feature]: https://img.shields.io/badge/feature-green.svg
[badge-enhancement]: https://img.shields.io/badge/enhancement-blue.svg
[badge-bugfix]: https://img.shields.io/badge/bugfix-purple.svg
[badge-experimental]: https://img.shields.io/badge/experimental-lightgrey.svg
[badge-maintenance]: https://img.shields.io/badge/maintenance-gray.svg
[badge-docs]: https://img.shields.io/badge/docs-orange.svg
