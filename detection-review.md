# Review of `src/asdex/detection/` — remaining findings

Scope: `detection/_api.py`, `detection/__init__.py`, and all 33 modules under `detection/_interpret/`,
read in full against the design principles in `CLAUDE.md` and the conventions in `_interpret/CLAUDE.md`.

This is a pruned revision of the original review.
The confirmed correctness bugs (C1–C6) and functional gaps (G1–G3) were fixed,
with regression tests pinning each one.
D1 was resolved by bundling the three state dicts into a `_PropState` dataclass
that every handler and state-touching helper takes uniformly.
D2 (list-aliasing rules and redundant deep copies) was resolved,
and the `dot_general` performance cliff was fixed by factoring propagation into row and column unions.
D5 (documenting the almost-everywhere derivative convention) was withdrawn as incorrect:
the piecewise-constant primitives have exactly-zero JVPs in JAX at every point,
and the kinked primitives (`abs`, `max`, `select_n`, `cond`) keep or union their dependencies,
so the detected pattern is a superset of the AD Jacobian for all inputs, including at kinks.
Original finding IDs are kept, which is why the numbering has gaps.

## Executive summary

No known correctness issues remain.
What is left is a set of minor convention violations (D3, D4)
and deduplication, simplification, and performance opportunities.

## Design observations

### D3. Factory-helper convention violations

`_interpret/CLAUDE.md` mandates the `_common` factory helpers so a backend swap touches one file.
Violations:

- `_api.py:173` builds raw sets: `[{col_offset + j} for j in range(size)]`.
- `_top_k.py:60` builds `[set() for ...]` instead of `_empty_index_sets`.

The `_api.py` case matters most since it is outside `_interpret` and easy to forget in a backend swap.

### D4. Structural inconsistencies (all minor)

- `_prop_iota` lives in `__init__.py` (`__init__.py:364`) while every other primitive has its own module,
  contradicting the documented structure. Move to `_iota.py`.
- `_linalg.py:6` imports `_common` by absolute path, every other module uses the relative `._common`.
- `_prop_qr` unpacks `q_var, r_var = eqn.outvars` (`_linalg.py:31`),
  which crashes with a bare unpacking error if `pivoting=True` adds a third outvar.
  A loud but uninformative failure, an explicit check with the report-issue message would fit the house style.
  It also uses `int(np.prod(...))` where `_numel` exists.
- `_run_prop` (`_api.py:191`) hand-builds the const dict with `strict=False`
  where `_seed_const_vals` (used everywhere else) exists and uses `strict=True`.
  Use the helper, the strictness inconsistency is a latent masking of length mismatches.
- The whole interpreter depends on `jax._src.core` and `jax._src.interpreters.partial_eval.dce_jaxpr`
  (`_api.py:11-12`), private APIs with no stability guarantee.
  Unavoidable for a jaxpr interpreter.
  A contract test now pins the `while_p` invars layout against the installed JAX
  (added alongside the C1 fix).
  Extending that idea to the other implicit layout contracts (params names, invars ordering elsewhere)
  would convert future JAX changes from silent wrong patterns into loud test failures.

## Deduplication opportunities

- **`_comparison.py` is 4 copies of one function.**
  `_prop_lt/_prop_le/_prop_gt/_prop_ge` (`_comparison.py:41,61,81,101`)
  differ only in the ufunc and the two bound comparisons.
  One `_prop_comparison(eqn, ..., ufunc, is_always_true, is_always_false)`
  plus four `functools.partial`s (or four thin wrappers) reduces ~90 lines to ~30
  and makes the always-true/always-false logic reviewable in one place.

- **`_stack.py` and `_concatenate.py` share their core.**
  Both pool input index sets, build offset position arrays per input,
  mirror the op with `np.stack`/`np.concatenate`, and permute
  (`_stack.py:56-68`, `_concatenate.py:41-50`).
  A shared helper parameterized by the numpy op (also covering the const propagation)
  removes one of the two implementations.

- **`_gather_flat_map` and `_scatter_flat_map` duplicate the index-vector iteration machinery.**
  `_gather.py:24-93` and `_scatter.py:22-124` are near-identical:
  batching-dims extraction, `si_batch_axes`, the nested `np.ndindex` loops,
  the `si_idx` construction, and the `start` assembly.
  A shared generator yielding `(batch_idx, si_batch_idx, start)` would leave only the
  genuinely different window/offset handling in each file.
  Also, `_gather.py:81` re-implements `_clamp_starts` (`_common.py:358`) inline.

- **Bounded-enumeration call sites repeat the same boilerplate.**
  Four handlers build `ranges` from `(lo, hi)` and wire up a `_make` closure the same way
  (`_gather.py:172`, `_scatter.py:245`, `_dynamic_slice.py:106,182`).
  A `_bounded_ranges(bounds)` helper for the
  `[range(int(lo), int(hi) + 1) ...]` construction is the cheap 80% win.

- **Broadcast position-mapping exists three times.**
  `_broadcast_to_output` (`_common.py:263`, value level),
  `_broadcast_flat` inside `_binary_elementwise` (`_elementwise.py:145`, index level),
  and the `in_coords` construction in `_prop_broadcast_in_dim` (`_broadcast.py:84-88`).
  One `_broadcast_flat_map(in_shape, out_shape) -> np.ndarray` covers all three
  (values become `val.ravel()[flat_map]`).
  Within `_broadcast.py` itself, the `intermediate_shape` computation is duplicated
  between the const path (lines 56-60) and the bounds path (lines 107-113).

- **`_atom_numel` re-implements `_numel(_atom_shape(atom))`** (`_common.py:152`).
  Two near-identical branches collapse to one line.

- **The report-an-issue message is inlined six times**
  (`__init__.py:129,413`, `_common.py:178,344`, `_reshape.py:48`, `_while.py:110`).
  A `_report_issue(msg)` helper in `_common` keeps the URL and phrasing in one place.

- **`_union_elementwise` TODO** (`_common.py:325`) already identifies that
  `select_n`'s dynamic path (`_select.py:55-61`) and parts of `_binary_elementwise` re-implement it.
  Worth doing, the select_n loop is exactly `_union_elementwise(case_indices, out_size)`.

## Simplification opportunities

- **`_prop_pad`** (`_pad.py:57-94`) walks every output element in Python with
  hand-rolled stride math (`_row_strides`/`_flat_to_coords`).
  The whole mapping is expressible with one strided-slice assignment on a position map:

  ```python
  flat_map = np.full(out_shape, -1)
  flat_map[tuple(slice(lo, lo + (n - 1) * (i + 1) + 1, i + 1) for ...)] = _position_map(in_shape)
  ```

  then `[pad_dep if m < 0 else in_indices[m] for m in flat_map.ravel()]`.
  This deletes the per-element loop, the early-break logic,
  and (together with a similar cleanup in `_conv`) the only users of
  `_row_strides`/`_flat_to_coords`, which could then be removed in favor of numpy built-ins.

- **`_prop_cumsum`** (`_cumsum.py:65-68`) re-unions the full prefix for every position,
  O(n²) set work with a large constant.
  A running union (`acc = acc | in_indices[p]`, storing `acc` per step) is simpler and cheaper,
  and sharing the intermediate sets is legal under the no-mutation invariant.

- **`_prop_scan`** computes `iter_length` from the xs shape with an `if xs else length` fallback
  (`_scan.py:76`) even though `params["length"]` is always present. Use `length` directly,
  and the two `AssertionError` shape checks become redundant.

- **`_api.jacobian_sparsity`** traces `f` twice:
  `jax.make_jaxpr` at `_api.py:63` and `jax.eval_shape` at `_api.py:65` purely to compute `m`.
  The output sizes are already on `closed_jaxpr.jaxpr.outvars[i].aval`.
  For expensive-to-trace functions this halves detection trace time.

- **`_fixed_point_loop`** (`_while.py:94-113`) uses `for ... else` with a `return` inside,
  so the `else` is equivalent to straight-line code after the loop.
  Moving the raise below the loop reads more directly
  (the `else` idiom signals a missing `break`, which never occurs here).

## Performance notes

Detection is a one-time cost, so none of these are urgent, but the cliffs are worth knowing:

- `_prop_conv_general_dilated` (`_conv.py:122-171`) and `_scatter_flat_map`
  run nested Python loops per output element (and per kernel/scatter position).
  For realistic CNN shapes this is minutes, not seconds. Vectorizing the flat-map construction with numpy
  (as gather already does for the slice extraction) is the fix when it becomes a problem.
- `_prop_scan` propagates the body once per timestep (`_scan.py:88`).
  A `lax.scan` with `length=100_000` runs 100k full jaxpr propagations.
  A fixed-point treatment (as in `while`) is the escape hatch when ys precision can be sacrificed,
  or a documented cap with a clear error.
- `_seed_const_vals` calls `np.asarray` on every closure constant (`_common.py:463`),
  copying e.g. all NN weights device-to-host and keeping them alive in `state.consts` for the whole analysis.
  Consts are only ever consumed as indices, masks, or zero-skipping values,
  so lazily materializing (or size-capping) would bound memory.
- `_conv.py:171` guards `if in_flat < len(lhs_indices)` and silently skips otherwise.
  If the coordinate math were ever wrong this hides it, contra "favor exceptions over wrong results".
  It should be impossible, so make it an assertion.

## What is working well

Worth keeping and worth imitating in new handlers:

- The dispatch table with derivative comments per primitive group (`__init__.py:161-360`)
  is an unusually readable inventory of semantic decisions.
- The unknown-primitive path raises instead of guessing (`_prop_throw_error`),
  and the conservative fallback list is explicit and short.
- The handler docstring template (semantics, math, example, jaxpr layout, URL) is applied consistently
  and makes each handler independently reviewable.
- Zero-size early returns are present in the handlers that need them
  (gather, broadcast, stack, unstack, sort, cumsum, qr).
- `_fixed_point_loop`'s monotone-lattice convergence argument is documented where the loop lives.
- Determinism is handled once, centrally (`_coo_from_index_sets` sorts columns, `_api.py:201`).
- DCE with `instantiate=True` to preserve input alignment (`_api.py:133-138`) is subtle and correctly explained.

## Suggested order of attack

1. Dedup batch: comparisons, stack/concatenate, gather/scatter iteration, bounded-enumeration ranges, broadcast maps.
2. D3 and D4 convention cleanups.
3. The simplification and performance items as they become relevant.
