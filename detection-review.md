# Review of `src/asdex/detection/` — remaining findings

Scope: `detection/_api.py`, `detection/__init__.py`, and all modules under `detection/_interpret/`,
read in full against the design principles in `CLAUDE.md` and the conventions in `_interpret/CLAUDE.md`.

This is a pruned revision of the original review.
The confirmed correctness bugs (C1–C6) and functional gaps (G1–G3) were fixed,
with regression tests pinning each one.
D1 (state bundling into `_PropState`), D2 (list-aliasing rules),
D3 (factory-helper violations), and D5 (withdrawn as incorrect) are resolved.
The deduplication batch is done:
comparisons share one core, stack/concatenate share a join core,
gather/scatter share the index-vector iteration,
bounded-enumeration call sites use `_bounded_ranges`,
broadcasting has a single implementation (`_broadcast_flat_map`),
and the report-issue message lives in `_report_issue`.
The simplification batch is done:
`_prop_pad` is vectorized, `_prop_cumsum` uses a running union,
`_prop_scan` reads `params["length"]` directly,
`jacobian_sparsity` no longer traces twice,
and the conv OOB guard is an assertion.
D4 is mitigated as far as it can be:
`tests/_interpret/test_jax_contracts.py` pins the implicit layout contracts.
The performance batch is done:
conv propagation is vectorized and factored through the union's associativity,
scan stops early once the carry saturates,
and closure constants are materialized lazily.
Original finding IDs are kept, which is why the numbering has gaps.

## Executive summary

No known correctness issues remain.
What is left is one structural dependency (D4) that is mitigated but cannot be removed,
and one residual performance limit that is inherent to exact per-timestep scan patterns.

## Design observations

### D4. Private JAX API dependency (mitigated, cannot remove)

The whole interpreter depends on `jax._src.core` and
`jax._src.interpreters.partial_eval.dce_jaxpr` (`_api.py:10-11`),
private APIs with no stability guarantee.
Unavoidable for a jaxpr interpreter.
The mitigation is in place:
`tests/_interpret/test_jax_contracts.py` pins every implicit layout contract
the handlers read against the installed JAX,
covering the while/scan/cond invars and inner-jaxpr layouts,
select_n case ordering, dynamic slice/update start ordering,
gather/scatter/conv/dot_general dimension numbers,
the pad `padding_config` convention, top_k output ordering,
iota `dimension` semantics, nested-jaxpr param alignment,
and `dce_jaxpr` input preservation under `instantiate=True`.
A future JAX change to any of these now fails loudly in that file
instead of surfacing as a silently wrong pattern.
What remains irreducible is the import dependency itself.

## Performance notes

The three cliffs from the original review are fixed:

- `_prop_conv_general_dilated` no longer loops per output element.
  The window map is built with numpy,
  and because set union is associative and commutative,
  channels are pre-unioned once per input spatial position,
  windows once per output spatial position,
  and all output channels of a feature/batch group alias the resulting sets.
  This removed `_flat_to_coords`, the last coordinate-level helper.
- `_prop_scan` detects carry saturation:
  when no xs slice carries input dependencies,
  the simulation stops once the carry index sets repeat between consecutive steps
  and replicates the last ys slice, staying exact.
  What remains irreducible: a scan whose xs carry distinct per-timestep dependencies
  (e.g. cumsum over the input) still runs one body propagation per timestep,
  which is the price of an exact per-timestep ys pattern.
- `_seed_const_vals` stores closure constants unconverted;
  `_atom_const_val` materializes to numpy on first read and caches.
  Never-read constants (e.g. conv kernels) are no longer copied device-to-host,
  and `_forward_into_jaxpr` forwards consts without materializing them.

## What is working well

Worth keeping and worth imitating in new handlers:

- The dispatch table with derivative comments per primitive group (`__init__.py:158-360`)
  is an unusually readable inventory of semantic decisions.
- The unknown-primitive path raises instead of guessing (`_prop_throw_error`),
  and the conservative fallback list is explicit and short.
- The handler docstring template (semantics, math, example, jaxpr layout, URL) is applied consistently
  and makes each handler independently reviewable.
- Zero-size early returns are present in the handlers that need them
  (gather, broadcast, unstack, sort, cumsum, qr).
- `_fixed_point_loop`'s monotone-lattice convergence argument is documented where the loop lives.
- Determinism is handled once, centrally (`_coo_from_index_sets` sorts columns, `_api.py:204`).
- DCE with `instantiate=True` to preserve input alignment (`_api.py:135-144`) is subtle and correctly explained.

## Suggested order of attack

Nothing open.
Revisit the per-timestep scan cost only if a workload
with input-dependent xs and very large `length` shows up.
