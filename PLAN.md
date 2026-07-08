# Review findings on `ah/fable-detection-fixes`

Scope: `git diff main...HEAD` (52 files, +2933/−1372),
covering the detection interpreter rework, its tests, and the CLAUDE.md rewrites.
Findings come from a multi-agent review at high effort.
Every finding below was independently re-verified against the code on this branch,
with verdicts CONFIRMED (reproduced from the source) or PLAUSIBLE (code facts confirmed, impact judgment).
Line numbers refer to this branch at commit `afd9d02`.
A scrutiny pass on 2026-07-08 re-verified all nine findings against the source
and folded implementation caveats into the fix sections below.

Work the findings top to bottom, they are ordered by severity.
T1 is a must-fix before merge.
P1 to P3 partially undo this branch's own performance goal (lazy closure constants) and should be fixed.
D1 to D5 are optional polish.

Conventions for whoever picks this up:

- Run `uv run ruff check --fix .`, `uv run ruff format .`, and `uv run ty check` before `uv run pytest`.
- Create a new commit per finding (or per coherent batch), never amend.
- Use Conventional Commits (`test:`, `perf:`, `refactor:`, `docs:`).
- Semantic line breaks in all prose (docstrings, comments, markdown).
- After fixing a finding, update its section here: mark it fixed and note the commit.
  Delete this file once every finding is resolved or explicitly rejected.

## T1. `test_stop_gradient` was accidentally destroyed (must fix)

**Location**: `tests/_interpret/test_internals.py:153`
**Status**: open

The branch deleted the `@pytest.mark.elementwise` marker and the `def test_stop_gradient():` line,
but left the test's docstring and body in place.
Both got grafted onto the tail of the unrelated `test_forward_into_jaxpr_preserves_laziness`,
where the docstring now sits as a stray string literal mid-function (line 153):

```python
    _forward_into_jaxpr(state, [outer], [inner])
    assert state.consts[inner] is closed.consts[0]  # still unconverted
    """stop_gradient passes dependencies through unchanged."""

    def f(x):
        return jax.lax.stop_gradient(x)

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    ...
```

The standalone stop_gradient regression test no longer exists.
A future regression in the stop_gradient handler would surface as a confusing failure
inside the const-laziness test, or not at all when that test is deselected by marker.
This violates the golden rule in `tests/CLAUDE.md`: never remove a test.

**Fix**:
End `test_forward_into_jaxpr_preserves_laziness` after its final assert (line 152).
The `elementwise` marker and all imports the body needs are already present in the file,
so the restoration is a pure paste.
Restore the test exactly as it exists on `main` (`git show main:tests/_interpret/test_internals.py`, line 123):

```python
@pytest.mark.elementwise
def test_stop_gradient():
    """stop_gradient passes dependencies through unchanged."""

    def f(x):
        return jax.lax.stop_gradient(x)

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)
```

**Verify**: `uv run pytest tests/_interpret/test_internals.py -v` collects and passes both
`test_forward_into_jaxpr_preserves_laziness` and `test_stop_gradient`.

## P1. `_prop_closed_jaxpr` materializes consts at every nested-jaxpr boundary

**Location**: `src/asdex/detection/_interpret/__init__.py:153-155`
**Status**: open

When forwarding const values out of a nested jaxpr,
`_prop_closed_jaxpr` reads every outvar through `_atom_const_val`:

```python
        # Forward const values symmetrically to bounds,
        # so indices computed inside the nested jaxpr stay resolvable outside.
        val = _atom_const_val(inner_outvar, state)
        if val is not None:
            state.consts[outvar] = val
```

`_atom_const_val` (`_common.py:225-233`) materializes lazily seeded device arrays
to host numpy and caches the copy.
So a jit/pjit-wrapped model that returns or threads a large closure constant
(for example NN weights) through a nested jaxpr
gets that constant copied device-to-host and kept alive for the whole analysis,
even when no downstream handler ever reads its value.
This re-introduces exactly the cost that commit `8270474`
("perf: materialize closure constants lazily") removed,
and it contradicts the documented laziness contract in `_interpret/CLAUDE.md`
("never-read constants are never copied to host").

The inward direction already does this correctly.
`_forward_into_jaxpr` (`_common.py:495-521`) forwards without materializing:
Literals are converted with `np.asarray` (cheap, Literals are small),
tracked vars are aliased (`state.consts[inner] = state.consts[outer]`),
and bounds are forwarded in the same loop.

`_prop_closed_jaxpr` is also the only outward-forwarding boundary:
`_scan.py`, `_while.py`, and `_cond.py` forward inward only,
so the fix below closes the regression completely.

**Fix**:
`_forward_into_jaxpr` already implements the exact logic the outward direction needs,
since jaxpr outvars are also atoms (Literal or Var) being mapped to fresh outer vars.
Replace the manual bounds forwarding (lines 149-150) and the `_atom_const_val` forwarding (lines 151-155)
with a single call:

```python
    _forward_into_jaxpr(state, closed.outvars, eqn.outvars)
```

keeping only the `state.indices[outvar] = indices` assignment in the zip loop.
Rename the helper to direction-neutral wording
(for example `_forward_across_jaxpr_boundary`)
and update its docstring and the `_interpret/CLAUDE.md` utilities list to match.
The rename also touches the import sites in `_scan.py`, `_while.py`, and `_cond.py`,
and the test name `test_forward_into_jaxpr_preserves_laziness`.

**Verify**:
Add a laziness regression test next to `test_forward_into_jaxpr_preserves_laziness`,
but do not target the helper itself:
after the refactor the outward path is the same function
the existing laziness test already covers,
so an identity check on the helper guards nothing new.
Target `_prop_closed_jaxpr` instead:
build a pjit equation whose nested jaxpr returns a captured device-array const,
run the handler,
and assert the outer outvar's stored const `is` the original unconverted array,
mirroring `assert state.consts[var] is closed.consts[0]` from `test_seed_const_vals_is_lazy`.
Then run the full suite.

## P2. Bounds helpers read the second operand before checking the first

**Location** (five sites, same pattern):
`src/asdex/detection/_interpret/_comparison.py:29-31` (`_get_bounds`),
`src/asdex/detection/_interpret/_elementwise.py:185-187` (`_propagate_bounds_add`),
`src/asdex/detection/_interpret/_elementwise.py:199-201` (`_propagate_bounds_sub`),
`src/asdex/detection/_interpret/_mul.py:51-53` (`_propagate_bounds_mul`),
`src/asdex/detection/_interpret/_div.py:53-55` (`_propagate_bounds_div`)
**Status**: open

All five helpers follow this shape:

```python
    b1 = _atom_value_bounds(eqn.invars[0], state)
    b2 = _atom_value_bounds(eqn.invars[1], state)
    if b1 is None or b2 is None:
        return ...
```

`_atom_value_bounds` calls `_atom_const_val`,
which materializes a lazily seeded const to host numpy and caches it.
For `x < big_captured_const` (or `x + c`, `x - c`, `x * c`, `x / c`) with traced `x`,
`b1` is `None` so the result is discarded,
yet `b2` was already evaluated and the constant copied device-to-host
and kept alive for the whole analysis.

`_propagate_const_binary` (`_common.py:283-289`) already solves this two functions away,
with an explicit early return and a comment
("Skip reading the second operand, so an input-dependent operand does not force materializing a large const").

Two scrutiny notes.
First, the five sites are the complete list:
every other `_atom_value_bounds` call site in the package is single-operand.
Second, the `_mul.py` site is consistency-only:
`_prop_mul` calls `_clear_where_zero(eqn, state, 1)` two lines earlier (`_mul.py:38`),
which unconditionally materializes operand 1's const for zero-skipping (`_common.py:310`),
so `x * c` pays the device-to-host copy regardless of the bounds reorder.
The real laziness wins are add and sub (every `x + bias` in an NN forward pass),
div (only operand 0 is zero-cleared),
and the comparisons.

**Fix**: in each of the five helpers, check `b1` before reading `b2`:

```python
    b1 = _atom_value_bounds(eqn.invars[0], state)
    if b1 is None:
        return ...
    b2 = _atom_value_bounds(eqn.invars[1], state)
    if b2 is None:
        return ...
```

The observable results are unchanged, only the evaluation order shifts.
Add a short why-comment at one site (or reference the `_propagate_const_binary` comment)
so the ordering is not "simplified" back.

**Verify**: full test suite, plus a laziness test if cheap to add
(seed a device-array const, run one comparison against a traced operand,
assert the stored const `is` the original unconverted array).

## P3. `_prop_select_n` fetches case consts and bounds it can never use

**Location**: `src/asdex/detection/_interpret/_select.py:61` and `:68`
**Status**: open

```python
    # When all inputs are statically known, compute the concrete result
    # so state.consts tracking isn't broken by this op.
    case_vals = [_atom_const_val(c, state) for c in cases]
    if which_val is not None and all(v is not None for v in case_vals):
        state.consts[out_var] = np.choose(...)

    # Propagate value bounds.
    case_bounds = [_atom_value_bounds(c, state) for c in cases]
```

`case_vals` is computed before the `which_val is not None` guard.
For `jnp.where(traced_mask, x, big_captured_const)` the selector is dynamic
(`which_val is None`), so the const result can never be stored,
yet every case constant is materialized device-to-host for nothing.
`case_bounds` at line 68 has the same problem:
it reads bounds (and thereby consts) for all cases up front,
even when the first case already returns `None` and the merged-bounds path (line 80) must bail.

**Fix**:

- Guard the const path: only build `case_vals` when `which_val is not None`,
  then `state.consts[out_var] = np.choose(which_val, case_vals)` when all are known.
  The current `[v for v in case_vals if v is not None]` filter is logically redundant inside `all(...)`,
  but it narrows `list[np.ndarray | None]` for the type checker, and `all(...)` does not narrow.
  Restructure so narrowing survives
  (for example collect non-None values in a typed loop),
  do not just delete the filter and fail `uv run ty check`.
- Make the bounds path short-circuit.
  The const-boolean-predicate branch (lines 71-77) only needs the bounds of the branch it selects,
  so read just that one.
  When the selected branch's bounds are `None`, return without storing bounds
  instead of falling through to the merged branch.
  The observable result is identical
  (the merge requires all case bounds, including the selected one),
  but falling through would materialize the other branches' consts for nothing.
  The merged branch should collect bounds with an early break on the first `None`
  instead of the up-front list comprehension.

**Verify**: full test suite,
`tests/_interpret/test_select.py` covers both selector kinds.

## D1. Primitive list duplicated between dispatch and `_UNARY_CONST_UFUNCS`

**Location**: `src/asdex/detection/_interpret/_elementwise.py:71-77`
and `src/asdex/detection/_interpret/__init__.py:165-171`
**Status**: open (verdict PLAUSIBLE, judgment call)

The four primitive names `floor`, `ceil`, `sign`, `not` are maintained in two places:
the `_prop_zero_derivative_unary_const` dispatch case in `_prop_dispatch`,
and the keys of `_UNARY_CONST_UFUNCS`.
`_prop_zero_derivative_unary_const` looks up with `.get` and silently no-ops on a miss:

```python
    ufunc = _UNARY_CONST_UFUNCS.get(eqn.primitive.name)
    if ufunc is not None:
        _propagate_const_unary(eqn, state, ufunc)
```

A contributor adding a primitive to one side but not the other gets no error.
The const chain silently breaks and downstream gather/scatter falls back to a conservative dense pattern,
which is exactly the class of precision bug this branch fixed.

**Fix**:
the dispatch is a `match` statement, so membership cannot be derived from the dict.
Make the miss loud instead:
index with `_UNARY_CONST_UFUNCS[eqn.primitive.name]` so a missing key raises `KeyError`,
and add a comment on the dict that its keys must stay in sync with the dispatch case.
Note the docstring of `_prop_zero_derivative_unary_const` already names the four primitives,
keep that in sync too.
Do not extend the fix by analogy:
the binary sibling `_propagate_const` (`_elementwise.py:88`) uses the same `.get` no-op intentionally,
since `_BINARY_CONST_UFUNCS` deliberately covers only a subset of a broad dispatch group
and a miss there means "no const propagation for this primitive", not a desync.
Note the asymmetry in the new comment
so a future consistency sweep doesn't make the binary lookup loud too.

**Verify**: full test suite.

## D2. The bounded-enumeration merge block is copy-pasted four times

**Location**:
`src/asdex/detection/_interpret/_gather.py:222-224`,
`src/asdex/detection/_interpret/_scatter.py:236-238`,
`src/asdex/detection/_interpret/_dynamic_slice.py:120-122`,
`src/asdex/detection/_interpret/_dynamic_slice.py:201-203`
**Status**: open

The identical three-line block

```python
        if any(si_index_sets):
            combined_si = _union_all(si_index_sets)
            result = [iset | combined_si for iset in result]
```

appears verbatim after every `_enumerate_bounded_patterns` success path
(with `start_index_sets` as the variable name in `_dynamic_slice.py`).
It unions the index/start operand's own input dependencies into every enumerated pattern.
Four sites must be updated in lockstep when the merge rule changes,
and the next handler gaining bounded enumeration can silently miss it.

**Fix**: add a helper in `_common.py` next to `_enumerate_bounded_patterns` and `_bounded_ranges`,
for example:

```python
def _merge_index_dependencies(
    result: list[IndexSet], index_sets: list[IndexSet]
) -> list[IndexSet]:
    """Union the index operand's own index sets into every enumerated pattern."""
    if not any(index_sets):
        return result
    combined = _union_all(index_sets)
    return [iset | combined for iset in result]
```

Use it at all four sites and add it to the Common Utilities list in `_interpret/CLAUDE.md`.
Mind the aliasing rules: the helper builds new sets, which is fine,
but do not mutate `result` in place.

**Verify**: full test suite,
gather/scatter/dynamic_slice bounded-enumeration tests cover all four sites.

## D3. `_dot_general` re-implements its own `_fixed_base_positions`

**Location**: `src/asdex/detection/_interpret/_dot_general.py:225-230`,
helper at `:19-39`
**Status**: open

The contracting-offset loops

```python
    lhs_offsets = np.zeros(n_contract, dtype=np.int64)
    for i, d in enumerate(lhs_contract):
        lhs_offsets += contract_coords[i] * lhs_strides[d]
    rhs_offsets = np.zeros(n_contract, dtype=np.int64)
    for i, d in enumerate(rhs_contract):
        rhs_offsets += contract_coords[i] * rhs_strides[d]
```

are element-for-element the body of `_fixed_base_positions`,
defined at line 19 of the same file.
Contract sizes are equal pairwise by dot_general's contract
(`lhs_contract[i]` pairs with `rhs_contract[i]` and has equal size),
so the replacement is exact:

```python
    lhs_offsets = _fixed_base_positions(lhs_shape, lhs_contract, lhs_strides)
    rhs_offsets = _fixed_base_positions(rhs_shape, rhs_contract, rhs_strides)
```

`contract_coords`, `contract_sizes`, and `n_contract` (lines 218-224) then have no remaining users
and can all be deleted.
The `n_contract = len(const_offsets)` at line 73 is a separate local inside `_one_const_indices`,
keep it.
Preserve the explanatory comment about offset sharing (lines 215-217),
adapting it to the helper call.
The `_fixed_base_positions` docstring is written for batch/free dims,
extend it to mention the contracting-dims use instead of replacing the existing text.

**Verify**: full test suite, `tests/_interpret/test_dot_general.py`.

## D4. `_iter_si_starts` returns an entangled 3-tuple

**Location**: `src/asdex/detection/_interpret/_gather.py:29-84`,
callers at `_gather.py:113-122` and `_scatter.py:64-72`
**Status**: open

The shared helper bundles two independent concerns into one return value:
the batch-shape computation `(batching_shape, si_batch_shape)`
and the single-pass iterator of `(batch_idx, si_batch_idx, start)` triples.
Each caller discards a different half.
`_gather_flat_map` uses the shapes but ignores the per-item batch indices
(`for _, _, raw_start in starts`).
`_scatter_flat_map` uses the per-item batch indices but ignores the shapes
(`_, _, starts = _iter_si_starts(...)`).
Every reader must work out which channel each caller consumes,
and a future caller could plausibly misuse the single-pass iterator after inspecting the shapes.

**Fix**: split the shape computation from the iteration.
For example, a `_si_batch_shapes(concrete_indices, operand_shape, operand_batching_dims, si_batching_dims)`
helper returning `(batching_shape, si_batch_shape)`,
with `_iter_si_starts` calling it internally and returning only the iterator.
Gather calls both, scatter calls only the iterator.
Keep the docstring's OOB-policy note ("starts are not clamped") with the iterator.
One trade-off to weigh:
the split forces recomputing `si_batch_axes` and `index_vector_dim` in both helpers,
or a shape helper that returns three values,
so it partly trades one wart for another.
Fixing and rejecting are both defensible.
If rejected, mark this finding rejected here instead of deleting it.

**Verify**: full test suite,
gather and scatter tests including the batching-dims cases.

## D5. Forbidden "deps" terminology in a new docstring

**Location**: `src/asdex/detection/_interpret/_dot_general.py:47-48`
**Status**: open

The `_contract_union_sets` docstring reads:

```
    For lhs these are the row sets ``deps(lhs[b, i, :])``,
    for rhs the column sets ``deps(rhs[b, :, j])``.
```

`_interpret/CLAUDE.md` (rewritten on this same branch) forbids the term:
"Docstrings — avoid the term 'deps'; prefer 'index sets' or 'input index sets'".

**Fix**: rephrase using the sanctioned terminology, for example:
"For lhs these are the row sets, the unioned index sets of ``lhs[b, i, :]``.
For rhs the column sets, the unioned index sets of ``rhs[b, :, j]``."
Grep the diff for other new `deps` occurrences while at it:
`git diff main...HEAD | grep -n '^+.*deps'`.

**Verify**: `uv run ruff check --fix .` and a targeted grep, no test impact.

## Refuted candidates (do not re-file)

The review also raised and then refuted three plausible-looking regressions.
Recorded here so follow-up agents do not rediscover them:

- **`_scatter.py:211` removed `_check_no_index_sets` guard**:
  the precise path requires `_atom_const_val(indices_atom, state) is not None`,
  and a const-resolved atom provably has all-empty index sets,
  so no dependency can be silently dropped there.
- **`_dynamic_slice.py:94` same pattern for start atoms**:
  the static-starts path is only taken when every start resolved as a const,
  which again implies empty index sets.
- **`_broadcast.py:73` suspected IndexError for 0-d tracked values**:
  JAX's own rank contract on `broadcast_in_dim`
  (rank of operand must equal `len(broadcast_dimensions)`)
  makes the claimed trigger state unconstructible.

One caveat on coverage:
one of the four finder agents died on an API error mid-run
(the angle covering data-flow/aliasing correctness),
so that angle is less thoroughly covered than the others.
A follow-up `/code-review` pass after the fixes land would close that gap.
