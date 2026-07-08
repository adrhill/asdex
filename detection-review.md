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
Of D4, only the private-API exposure below remains.
Original finding IDs are kept, which is why the numbering has gaps.

## Executive summary

No known correctness issues remain.
What is left is one structural dependency (D4) that can only be mitigated, not removed,
and known performance cliffs that are acceptable until someone hits them.

## Design observations

### D4. Private JAX API dependency (mitigate, cannot remove)

The whole interpreter depends on `jax._src.core` and
`jax._src.interpreters.partial_eval.dce_jaxpr` (`_api.py:10-11`),
private APIs with no stability guarantee.
Unavoidable for a jaxpr interpreter.
A contract test pins the `while_p` invars layout against the installed JAX
(added alongside the C1 fix).
Extending that idea to the other implicit layout contracts
(params names, invars ordering elsewhere)
would convert future JAX changes from silent wrong patterns into loud test failures.

## Performance notes

Detection is a one-time cost, so none of these are urgent, but the cliffs are worth knowing:

- `_prop_conv_general_dilated` (`_conv.py:122-178`) runs nested Python loops
  per output element and kernel position.
  For realistic CNN shapes this is minutes, not seconds.
  Vectorizing the flat-map construction with numpy
  (as gather and pad already do) is the fix when it becomes a problem.
  That cleanup would also remove the last users of
  `_row_strides` at coordinate level and `_flat_to_coords` (`_common.py:435-461`).
- `_prop_scan` propagates the body once per timestep (`_scan.py:77`).
  A `lax.scan` with `length=100_000` runs 100k full jaxpr propagations.
  A fixed-point treatment (as in `while`) is the escape hatch when ys precision can be sacrificed,
  or a documented cap with a clear error.
- `_seed_const_vals` calls `np.asarray` on every closure constant (`_common.py:475`),
  copying e.g. all NN weights device-to-host and keeping them alive in `state.consts`
  for the whole analysis.
  Consts are only ever consumed as indices, masks, or zero-skipping values,
  so lazily materializing (or size-capping) would bound memory.

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

1. Contract tests for the remaining implicit JAX layout assumptions (D4).
2. The performance items as they become relevant.
