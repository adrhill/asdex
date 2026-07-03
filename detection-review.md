# Review of `src/asdex/detection/`

Scope: `detection/_api.py`, `detection/__init__.py`, and all 33 modules under `detection/_interpret/`,
read in full against the design principles in `CLAUDE.md` and the conventions in `_interpret/CLAUDE.md`.
Suspected correctness issues were verified empirically by comparing detected patterns
against dense `jax.jacobian` / `jax.jacfwd` reference Jacobians.
No code was changed.

## Executive summary

The detection subsystem is in good shape overall.
The handler-per-primitive structure, the factory-helper convention, the conservative-fallback invariants,
and the docstring discipline are all consistently applied and make the code easy to navigate.

The review found **two confirmed correctness bugs** that silently produce wrong sparsity patterns
(missing nonzeros, so decompression would return wrong Jacobians without any error):

1. `_prop_while` swaps cond consts and body consts when slicing `eqn.invars`.
2. Const propagation for integer `div` uses `np.divide` (true division),
   which diverges from `lax.div` truncation semantics.

It also found one confirmed usability bug (a misleading hard error for valid user code),
one confirmed precision gap (consts are not forwarded out of nested jaxprs),
and a set of design, deduplication, and simplification opportunities detailed below.

| ID | Severity | Finding |
|----|----------|---------|
| C1 | High | `_prop_while` mislabels cond consts as body consts, wrong pattern (verified) |
| C2 | High | Integer `div`/`rem` const propagation uses numpy semantics, wrong pattern (verified) |
| C3 | Medium | Scatter replace with duplicate static indices assumes last-writer-wins |
| C4 | Medium | Gather/scatter ignore the `mode` parameter (scatter + `mode='clip'` loses deps) |
| C5 | Medium | `_index_sets` silently invents a wrong-sized default for unknown vars |
| C6 | Low | `_propagate_bounds_div` floor semantics can produce bounds that exclude the true value |
| G1 | Medium | `_check_no_index_sets` hard-errors on valid programs with a misleading message (verified) |
| G2 | Medium | Consts are not forwarded out of nested jaxprs, needless conservative fallback (verified) |
| G3 | Low | `state_bounds` is not threaded into `while`/`cond`/`scan` bodies |

## Confirmed correctness bugs

### C1. `_prop_while` swaps cond consts and body consts (high)

`_while.py:52` slices the body consts from the front of `eqn.invars`:

```python
body_consts = eqn.invars[:body_nconsts]
```

But JAX binds the while primitive with **cond consts first**
(`jax/_src/lax/control_flow/loops.py:1687`):

```python
outs = while_p.bind(*cond_consts, *body_consts, *init_vals, ...)
```

So the true layout is `[cond_consts..., body_consts..., carry_init...]`,
and the docstring at `_while.py:36` documents the wrong order.
The carry slice at `_while.py:53` happens to be correct because it uses the sum of both counts.

Whenever `cond_nconsts > 0` (the loop condition closes over anything traced),
the handler seeds the body jaxpr's const inputs with the *cond* consts' index sets and values.
The body's real const dependencies are dropped from the pattern
and the cond const's dependencies leak in.
`_prop_jaxpr` zips with `strict=False` at `__init__.py:100`,
so even a length mismatch between the two const groups is silently absorbed
(unseeded body invars then fall into the `_index_sets` default, see C5).

Verified repro (missing nonzeros, silently wrong Jacobian downstream):

```python
def f(x):
    thresh = x[0] * 1.0                      # cond const
    step = x[1:3]                            # body const
    return lax.while_loop(lambda c: c[0] < thresh,
                          lambda c: c + step,
                          x[3:5] * 1.0)
# detected: {(0,0), (0,3), (1,0), (1,4)}
# actual:   {(0,1), (0,3), (1,2), (1,4)}   -> (0,1) and (1,2) are MISSING
```

Fix: slice `cond_consts = invars[:cond_nconsts]` and
`body_consts = invars[cond_nconsts : cond_nconsts + body_nconsts]`,
and correct the docstring.
A regression test with `cond_nconsts > 0` and `cond_nconsts != body_nconsts` would pin this down.
Existing tests presumably only cover conditions that close over nothing (`cond_nconsts == 0`),
which is why this never surfaced.

### C2. Integer `div` and `rem` const propagation uses numpy semantics (high)

Two sites map the `div` primitive to `np.divide`:
the ufunc table at `_elementwise.py:31` and `_prop_div` at `_div.py:40`.
`np.divide` is true division and returns floats,
while `lax.div` on integers truncates toward zero.
Any integer index arithmetic that flows through `div` before a gather/scatter
can therefore resolve to the wrong concrete indices.

Verified repro (const chain inside a cond branch, where invars are Vars and equations are not folded):

```python
def tb(ops):
    i, xx = ops                               # i = [0, 1, 2] forwarded const
    j = lax.mul(lax.div(i, jnp.int32(2)), jnp.int32(2))
    return xx[j] * 1.0                        # lax: j = [0,0,2]; np.divide chain: j = [0,1,2]
# detected row 1: {1}    actual row 1: {0}   -> (1,0) is MISSING
```

`rem` has the same class of problem at `_elementwise.py:36`:
`np.remainder` takes the sign of the divisor while `lax.rem` takes the sign of the dividend
(`np.remainder(-7, 3) == 2` but `lax.rem(-7, 3) == -1`).
In my repro the branch union happened to mask the wrong row, but the computed const is demonstrably wrong,
so the same silent-missing-deps failure is reachable.

Fix suggestions:
map `rem` to `np.fmod` (C-style, matches `lax.rem`),
and replace `np.divide` with a dtype-aware wrapper that truncates toward zero for integer inputs,
for example `np.trunc(np.true_divide(a, b)).astype(result_dtype)`.
Note also that `np.power` on integer consts raises for negative exponents,
so a defensive wrapper around const evaluation
(fall back to "no const" on numpy errors instead of crashing detection) may be worth considering.

### C3. Scatter replace with duplicate static indices assumes last-writer-wins (medium)

`_scatter.py:150-153` resolves duplicate targets under replace semantics by taking the last update.
XLA leaves the winner **implementation-defined** when `scatter` (replace) receives duplicate indices,
so the safe pattern is the union of all updates targeting the position
(optionally including the operand, since some backends may drop all writes).
Taking only the last writer can under-approximate on a backend that picks a different winner.
This is a one-line change in the `else` branch and only affects the already-rare duplicate-index case,
so unioning costs no precision in practice.

### C4. `mode` parameter of gather/scatter is ignored (medium)

Neither `_gather_flat_map` nor `_scatter_flat_map` reads `eqn.params["mode"]` (`GatherScatterMode`).

- Gather hardcodes clamping (`_gather.py:81-83`), which matches the default retrieval semantics.
  Under `mode='fill'` an out-of-bounds output is a constant fill value with no dependency,
  so the clamped pattern is a superset. Safe, only imprecise.
- Scatter hardcodes dropping out-of-bounds updates (`_scatter.py:89-94`),
  which matches the default `FILL_OR_DROP`.
  But under `mode='clip'` (reachable via `x.at[idx].set(v, mode='clip')`)
  the update really lands at the clamped position,
  and dropping it **loses a true dependency**. Silent wrong pattern.

Per the "favor exceptions over wrong results" principle,
the minimal fix is to read `mode` and raise (or go conservative) for non-default scatter modes.

### C5. `_index_sets` invents a wrong-sized default for unknown vars (medium)

`_common.py:149` returns `[_empty_index_set()]` when a `Var` is missing from `state_indices`:

```python
return state_indices.get(atom, [_empty_index_set()])
```

Every var should have been seeded (invars, constvars) or written by a handler,
so a miss means a handler bug upstream.
Returning a length-1 list of empty sets both guesses (dependencies vanish)
and gets the length wrong (downstream `len(in_indices)`-based logic misbehaves).
This is exactly the guessing that `CLAUDE.md` says to avoid,
and it is what would absorb the misalignment in C1 into silence.
Recommendation: raise a `KeyError`-style internal error with the report-an-issue message.
Literals keep the existing synthesized-empty-sets path, which is correct.

### C6. `_propagate_bounds_div` floor semantics for integers (low)

`_div.py:69` uses `np.floor_divide` for integer dtypes.
`lax.div` truncates toward zero, and for negative intervals floor and truncation differ,
so a propagated bound can exclude the value the program actually computes
(for bounds `[-5, -5] / [2, 2]`: propagated `(-3, -3)`, actual `lax.div` result `-2`).
Bounded enumeration would then never try the true index.
Negative index bounds are exotic (bounds currently originate from argmax, which is nonnegative),
hence low severity, but the semantics should match `lax.div` for the same reason as C2.

Related edge: `_propagate_bounds_integer_pow` (`_elementwise.py:354`) treats every odd exponent as monotone.
For negative odd exponents that is false and produces inverted `(lo, hi)`.
Probably unreachable from index arithmetic, but worth an explicit `y < 0` bail-out.

## Functional gaps

### G1. Hard error with a misleading message for input-dependent auxiliary inputs (medium, verified)

`_prop_scatter` (`_scatter.py:207`), `_prop_dynamic_slice`/`_prop_dynamic_update_slice`
(`_dynamic_slice.py:94,167`), and `_prop_conv_general_dilated` (`_conv.py:55`)
call `_check_no_index_sets` and **raise** when the index/kernel input depends on the function inputs.
This fires on perfectly valid user programs:

```python
def f(x):
    return x.at[jnp.argsort(x)].set(x * 2.0)
# ValueError: 'scatter' handler assumes an auxiliary input has no dependency ...
# "Please help out asdex's development by reporting this at ..."
```

Two problems:

1. A correct conservative answer exists
   (all outputs depend on operand + updates + index deps, exactly what `_prop_gather` does in this case),
   so raising is stricter than the design principle requires.
   Gather already demonstrates the graceful pattern in the same situation, the four handlers are inconsistent.
2. The message tells the user they found an asdex bug and should file an issue.
   For a known, documented limitation (the TODOs at `_scatter.py:206`, `_dynamic_slice.py:92,165`, `_conv.py:54`)
   this is the wrong message. If raising is kept, the message should describe the limitation instead.

The same applies to a data-dependent convolution kernel
(for example a bilinear model `conv(x, g(x))`), which currently raises.

### G2. Consts are not forwarded out of nested jaxprs (medium, verified)

`_prop_closed_jaxpr` (`__init__.py:146-154`) forwards `state_bounds` from inner outvars to outer outvars
but does **not** forward `state_consts`.
A constant index array computed inside a `jit`-wrapped call and used by an outer gather
therefore falls back to conservative:

```python
def f(x):
    @jax.jit
    def make(i, xx):
        return i * 2, xx * 1.0
    idx, x2 = make(jnp.array([1, 0, 2]), x)
    return x2[jnp.floor_divide(idx, 2)]
# detected: fully dense 3x3    actual: 3 nonzeros
```

Adding a symmetric const forward in the same loop
(mirroring the two lines that forward bounds, using `_atom_const_val` so Literal outvars also work)
would close the gap.
Note that fixing this widens the reach of C2, another reason to fix C2 first.

### G3. `state_bounds` stops at `while`/`cond`/`scan` boundaries (low)

`_prop_dispatch` passes `state_bounds` into `_prop_closed_jaxpr` for `pjit`/`custom_jvp_call`/etc.,
but `_prop_while`, `_prop_cond`, and `_prop_scan` receive only `state_consts`
and call `_prop_jaxpr(body, inputs, state_consts)` with no bounds
(`_while.py:71`, `_cond.py:53`, `_scan.py:100`).
An argmax-derived index used inside a loop body or branch therefore degrades to conservative.
This is safe but inconsistent with the closed-jaxpr path.
The stale `PropJaxprFn` alias (`_common.py:57-60`) still declares the 3-argument signature
even though `_prop_jaxpr` grew a fourth parameter,
which is how the omission stays invisible to the type checker.
See also D1, which would fix this class of drift structurally.

## Design observations

### D1. Three parallel state dicts invite signature drift

`state_indices`, `state_consts`, and `state_bounds` are threaded separately through every handler.
The cost is visible in the code today:

- handlers have four different signatures
  (some take 2, 3, or 4 states, some with `| None = None` defaults such as
  `_prop_integer_pow` at `_elementwise.py:305` and `_prop_convert_element_type` at `_elementwise.py:413`,
  even though `_prop_dispatch` always passes everything),
- `PropJaxprFn` is stale (G3),
- forwarding into nested jaxprs needs two near-identical helpers
  (`_forward_const_vals` at `_common.py:445` and `_forward_value_bounds` at `_common.py:433`),
  and each call site must remember to call both (while/cond/scan currently forget bounds),
- `_prop_closed_jaxpr` forwards bounds out but forgot consts (G2).

Bundling the three dicts into one `_PropState` dataclass passed uniformly would collapse the signatures,
make "forward everything into the inner jaxpr" a single function that cannot be half-applied,
and remove the optional-parameter noise.
This is squarely the "minimize complexity" and "information hiding" goals of `CLAUDE.md`,
and it prevents the entire G2/G3 bug class rather than patching instances.

### D2. Aliasing rules are only documented for sets, not lists

`_interpret/CLAUDE.md` documents that index *sets* are shared and must never be mutated.
But whole *lists* are shared too:
`_prop_squeeze` (`_squeeze.py:31`) and `_prop_reshape` (`_reshape.py:66`) alias the input list object,
while `_clear_where_zero` (`_common.py:264-266`) mutates a list in place by replacing entries.
This is currently safe only because `_clear_where_zero` is applied exclusively to lists
freshly created by `_binary_elementwise`/`_copy_index_sets` in the same handler.
That invariant is real but undocumented and fragile.
Either document it next to the set-aliasing rule, or make `_clear_where_zero` build a new list.

Relatedly, the pass-through handlers are inconsistent about copying:
`_prop_unary_elementwise` and `_prop_convert_element_type` **deep-copy** every set
(`_copy_index_sets`, `_elementwise.py:408,441`),
while `_prop_squeeze` shares the list outright.
Under the never-mutate invariant the deep copies are unnecessary,
and they are O(total nnz) per elementwise op,
a real cost on chains like `exp(sin(tanh(x)))` over large sparse inputs.
Pick one convention (sharing is cheaper and already proven safe by squeeze/reshape).

### D3. Factory-helper convention violations

`_interpret/CLAUDE.md` mandates the `_common` factory helpers so a backend swap touches one file.
Violations:

- `_api.py:173` builds raw sets: `[{col_offset + j} for j in range(size)]`.
- `_top_k.py:60` builds `[set() for ...]` instead of `_empty_index_sets`.
- `_select.py:62` uses `_empty_index_set()` correctly but then relies on `|=` accumulation,
  fine, listed only for completeness.
- `_while.py:95-96` reimplements `_copy_index_sets` inline (`[s.copy() for s in carry[i]]`).

The `_api.py` case matters most since it is outside `_interpret` and easy to forget in a backend swap.

### D4. Structural inconsistencies (all minor)

- `_prop_iota` lives in `__init__.py` (`__init__.py:366`) while every other primitive has its own module,
  contradicting the documented structure. Move to `_iota.py`.
- `_linalg.py:6` imports `_common` by absolute path, every other module uses the relative `._common`.
- `_prop_qr` unpacks `q_var, r_var = eqn.outvars` (`_linalg.py:31`),
  which crashes with a bare unpacking error if `pivoting=True` adds a third outvar.
  A loud but uninformative failure, an explicit check with the report-issue message would fit the house style.
  It also uses `int(np.prod(...))` where `_numel` exists.
- `_run_prop` (`_api.py:188-191`) hand-builds the const dict with `strict=False`
  where `_seed_const_vals` (used everywhere else) exists and uses `strict=True`.
  Use the helper, the strictness inconsistency is a latent masking of length mismatches.
- The whole interpreter depends on `jax._src.core` and `jax._src.interpreters.partial_eval.dce_jaxpr`
  (`_api.py:11-12`), private APIs with no stability guarantee.
  Unavoidable for a jaxpr interpreter, but C1 shows how implicit layout contracts
  (invars ordering, params names) rot silently.
  A small compatibility test module that asserts these contracts against the installed JAX
  (for example, trace a while loop with known consts and assert the invars order)
  would convert future JAX changes from silent wrong patterns into loud test failures.

### D5. The a.e.-zero-derivative convention deserves user-facing documentation

Comparisons, `argmax`, `floor`, branch predicates (`cond`, `select_n`),
and the `while` termination condition all propagate **no** dependencies,
consistently implementing "Jacobian almost everywhere".
This is the right convention and is applied uniformly,
but it means the "valid for all inputs" claim in the `jacobian_sparsity` docstring (`_api.py:38-40`)
holds only off the measure-zero set where these primitives kink.
Worth one sentence in the public docstring, since users of `check_jacobian_correctness`
will occasionally sit exactly on a kink (for example `x == 0` with `abs`).

## Deduplication opportunities

- **`_comparison.py` is 4 copies of one function.**
  `_prop_lt/_prop_le/_prop_gt/_prop_ge` differ only in the ufunc and the two bound comparisons.
  One `_prop_comparison(eqn, ..., ufunc, is_always_true, is_always_false)`
  plus four `functools.partial`s (or four thin wrappers) reduces ~90 lines to ~30
  and makes the always-true/always-false logic reviewable in one place.

- **`_stack.py` and `_concatenate.py` share their core.**
  Both pool input index sets, build offset position arrays per input,
  mirror the op with `np.stack`/`np.concatenate`, and permute
  (`_stack.py:56-71`, `_concatenate.py:42-53`).
  A shared helper parameterized by the numpy op (also covering the const propagation)
  removes one of the two implementations.

- **`_gather_flat_map` and `_scatter_flat_map` duplicate the index-vector iteration machinery.**
  `_gather.py:44-79` and `_scatter.py:41-79` are near-identical:
  batching-dims extraction, `si_batch_axes`, the nested `np.ndindex` loops,
  the `si_idx` construction, and the `start` assembly.
  A shared generator yielding `(batch_idx, si_batch_idx, start)` would leave only the
  genuinely different window/offset handling in each file.
  Also, `_gather.py:81-83` re-implements `_clamp_starts` (`_common.py:325`) inline.

- **Bounded-enumeration call sites repeat the same boilerplate.**
  Four handlers build `ranges` from `(lo, hi)` and wire up a `_make` closure the same way
  (`_gather.py:170-186`, `_scatter.py:221-241`, `_dynamic_slice.py:107-123,180-200`).
  A `_bounded_ranges(bounds)` helper for the
  `[range(int(lo), int(hi) + 1) ...]` construction is the cheap 80% win.

- **Broadcast position-mapping exists three times.**
  `_broadcast_to_output` (`_common.py:231`, value level),
  `_broadcast_flat` inside `_binary_elementwise` (`_elementwise.py:114`, index level),
  and the `in_coords` construction in `_prop_broadcast_in_dim` (`_broadcast.py:88-93`).
  One `_broadcast_flat_map(in_shape, out_shape) -> np.ndarray` covers all three
  (values become `val.ravel()[flat_map]`).
  Within `_broadcast.py` itself, the `intermediate_shape` computation is duplicated
  between the const path (lines 60-65) and the bounds path (lines 114-121).

- **`_atom_numel` re-implements `_numel(_atom_shape(atom))`** (`_common.py:126-139`).
  Two near-identical branches collapse to one line.

- **The report-an-issue message is inlined five times**
  (`__init__.py:129-134,415-419`, `_common.py:307-312`, `_reshape.py:48-53`, `_while.py:112-117`).
  A `_report_issue(msg)` helper in `_common` keeps the URL and phrasing in one place.

- **`_union_elementwise` TODO** (`_common.py:291`) already identifies that
  `select_n`'s dynamic path (`_select.py:60-65`) and parts of `_binary_elementwise` re-implement it.
  Worth doing, the select_n loop is exactly `_union_elementwise(case_indices, out_size)`.

## Simplification opportunities

- **`_prop_pad`** (`_pad.py:64-94`) walks every output element in Python with
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
  (`_scan.py:80`) even though `params["length"]` is always present. Use `length` directly,
  and the two `AssertionError` shape checks become redundant.

- **`_api.jacobian_sparsity`** traces `f` twice:
  `jax.make_jaxpr` at `_api.py:63` and `jax.eval_shape` at `_api.py:65` purely to compute `m`.
  The output sizes are already on `closed_jaxpr.jaxpr.outvars[i].aval`.
  For expensive-to-trace functions this halves detection trace time.

- **`_fixed_point_loop`** (`_while.py:98-118`) uses `for ... else` with a `return` inside,
  so the `else` is equivalent to straight-line code after the loop.
  Moving the raise below the loop reads more directly
  (the `else` idiom signals a missing `break`, which never occurs here).

- **`_prop_dot_general`** carries a stale workaround:
  the `np.broadcast_to(...)` calls at `_dot_general.py:151-156` are no-ops now that
  every coordinate entry is built with `out_shape`-sized arrays (`lhs_fixed` values or `np.full(out_shape, ...)`),
  and the comment describes a shape situation that can no longer arise.

## Performance notes

Detection is a one-time cost, so none of these are urgent, but the cliffs are worth knowing:

- `_prop_conv_general_dilated` (`_conv.py:112-164`), `_prop_dot_general` (`_dot_general.py:134-166`),
  and `_scatter_flat_map` run nested Python loops per output element (and per kernel/contraction position).
  For realistic CNN shapes this is minutes, not seconds. Vectorizing the flat-map construction with numpy
  (as gather already does for the slice extraction) is the fix when it becomes a problem.
- `_prop_scan` propagates the body once per timestep (`_scan.py:93`).
  A `lax.scan` with `length=100_000` runs 100k full jaxpr propagations.
  A fixed-point treatment (as in `while`) is the escape hatch when ys precision can be sacrificed,
  or a documented cap with a clear error.
- `_seed_const_vals` calls `np.asarray` on every closure constant (`_common.py:429-430`),
  copying e.g. all NN weights device-to-host and keeping them alive in `state_consts` for the whole analysis.
  Consts are only ever consumed as indices, masks, or zero-skipping values,
  so lazily materializing (or size-capping) would bound memory.
- The deep copies in unary elementwise handlers (see D2) are the cheapest win.
- `_conv.py:161-162` guards `if in_flat < len(lhs_indices)` and silently skips otherwise.
  If the coordinate math were ever wrong this hides it, contra "favor exceptions over wrong results".
  It should be impossible, so make it an assertion.

## What is working well

Worth keeping and worth imitating in new handlers:

- The dispatch table with derivative comments per primitive group (`__init__.py:157-363`)
  is an unusually readable inventory of semantic decisions.
- The unknown-primitive path raises instead of guessing (`_prop_throw_error`),
  and the conservative fallback list is explicit and short.
- The handler docstring template (semantics, math, example, jaxpr layout, URL) is applied consistently
  and makes each handler independently reviewable.
- Zero-size early returns are present in the handlers that need them
  (gather, broadcast, stack, unstack, sort, cumsum, qr).
- `_fixed_point_loop`'s monotone-lattice convergence argument is documented where the loop lives.
- Determinism is handled once, centrally (`_coo_from_index_sets` sorts columns, `_api.py:199-215`).
- DCE with `instantiate=True` to preserve input alignment (`_api.py:130-139`) is subtle and correctly explained.

## Suggested order of attack

1. **C1** (while consts swap): two-line fix plus docstring, add a `cond_nconsts > 0` regression test.
2. **C2** (div/rem consts): dtype-aware ufuncs in the two `div` sites and the `rem` table entry, tests via cond-branch const chains.
3. **C5** (`_index_sets` default): turn the guess into an error, then re-run the suite. This may surface latent handler gaps.
4. **G1** (misleading hard error): conservative fallback (mirroring gather) or an honest error message.
5. **G2** (const forwarding out of nested jaxprs): two lines in `_prop_closed_jaxpr`, after C2.
6. **D1** (`_PropState` bundle): structural fix that retires G3 and the stale `PropJaxprFn`.
7. Dedup batch: comparisons, stack/concatenate, gather/scatter iteration, bounded-enumeration ranges, broadcast maps.
8. C3, C4, C6 and the simplification/performance items as they become relevant.
