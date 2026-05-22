# asdex

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
