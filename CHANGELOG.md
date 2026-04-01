# asdex

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
