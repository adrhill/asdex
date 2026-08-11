# _interpret — Custom Jaxpr Interpreter for Index Set Propagation

Implements a custom jaxpr interpreter that propagates per-element dependency index sets (`set[int]`)
through primitives to determine Jacobian sparsity patterns.

## Structure

- `__init__.py` — `_prop_jaxpr`, `_prop_dispatch`, fallback handling.
- `_common.py` — shared types (`IndexSet`, `_PropState`) and utilities.
- Each JAX primitive has its own module: `_foo.py` contains `_prop_foo`.
  Includes `_cumsum.py` for cumulative sum.
- Handlers for external packages (Equinox, Flax, etc.) live in their own subfolders
  (e.g., `_equinox/`).

## Key Types

- `IndexSet` = `set[int]` — a single per-element dependency set
- `list[IndexSet]` — per-element dependency sets for one array
- `StateIndices` = `dict[Var, list[IndexSet]]` — maps jaxpr variables to their index sets
- `StateConsts` = `dict[Var, ArrayLike]` — statically-known values for precise gather/scatter.
  Seeded closure constants stay in their original array type
  and are materialized to numpy by `_atom_const_val` on first read,
  so never-read constants (e.g. conv kernels) are never copied to host.
- `ValueBounds` = `tuple[np.ndarray, np.ndarray]` — per-element inclusive (lo, hi) bounds for one array
- `StateBounds` = `dict[Var, ValueBounds]` — maps jaxpr variables to their value bounds
- `_PropState` — bundles the three dicts above as `state.indices`, `state.consts`, and `state.bounds`.
  Every handler takes `(eqn, state)`,
  and every `_common` helper that touches state takes the whole bundle,
  so adding a new state component never changes signatures again.
  `indices` is scoped to a single jaxpr
  (each nested jaxpr gets a fresh dict via `_prop_jaxpr`, so intermediates can be freed),
  while `consts` and `bounds` are shared across nested scopes by aliasing.

## Naming Conventions

**Terminology** — "indices" and "map" mean different things:
- **"indices" / "index sets"**: `list[IndexSet]`,
  the per-element dependency sets used for sparsity tracking.
- **"map"**: numpy integer arrays that map output positions to input positions.
  Not index sets.

**Construction** — always use the factory helpers from `_common`:
- `_empty_index_set()` instead of `set()`
- `_singleton_index_set(i)` instead of `{i}`
- `_empty_index_sets(n)` instead of `[set() for _ in range(n)]`
- `_identity_index_sets(n)` instead of `[{i} for i in range(n)]`

This ensures a future backend swap only requires changing the helpers,
not every handler.
See [Backend Surface](#backend-surface) for the full set of operations
a handler may rely on.

**Variable names** — use these consistently across handlers:
- `in_indices`: input index sets (from `_index_sets(state, atom)`)
- `in_shape`: input array shape (from `_atom_shape(atom)`)
- `in_val`: const value for a unary input (from `_atom_const_val(atom, state)`)
- `in1_val` / `in2_val`: const values for binary inputs.
  Use descriptive prefixes when roles differ:
  `lhs_val` / `rhs_val` (dot_general), `pred_val` / `which_val` (select), etc.
- `in_bounds` / `in1_bounds` / `in2_bounds`: value bounds for inputs
  (from `_atom_value_bounds(atom, state)`)
- `flat_map`: a flat integer array mapping output positions to input positions

**Docstrings** — avoid the term "deps"; prefer "index sets" or "input index sets".

## Backend Surface

`IndexSet` is currently `set[int]`.
`pyroaring.BitMap` and int bitmasks were benchmarked and lost
for the typical workload of small sparse sets over a large universe
(see the `IndexSet` docstring in `_common.py`).
The seam is kept intact anyway so the choice can be revisited cheaply.

Handlers may rely on exactly these operations, and nothing else:

- **Construction** through the four factory helpers above.
  Never `set()` or `{i}`.
- **Union** through `a | b`, `a |= b`, `_union_all`, or `_union_elementwise`.
  Both operators are safe to use directly,
  since every candidate backend supports them.
- **Emptiness** through truthiness: `if not iset`, `any(sets)`.
  Every candidate backend reports emptiness through `__len__` or `__bool__`.
- **Copying** through `_copy_index_set` (one set) or `_copy_index_sets` (a whole list).
  Never call `.copy()` on an index set directly.
  A handler that accumulates with `|=` must copy first,
  since sets in `state.indices` are shared (see [Index Set Aliasing](#index-set-aliasing)).
  Prefer copy-then-accumulate over collecting operands into a list for `_union_all`:
  the list costs a temporary allocation per output element,
  which measurably dominates when index sets are small.
- **Annotations** always spell `IndexSet` or `list[IndexSet]`, never `set[int]`.

Iterating an index set to recover concrete integers
happens in exactly one place,
`_coo_from_index_sets` in `detection/_api.py`,
which sorts each set when building the COO pattern.
Keeping that the only iteration site
is what lets a backend without cheap ordered iteration stay viable.

## Common Utilities in `_common.py`

- **`_position_map(shape)`** —
  builds an array where each element holds its own flat position.
  Applying operations (transpose, slice, flip) to this array
  reveals which input position each output position reads from.
- **`_permute_indices(in_indices, flat_map)`** —
  builds output index sets by looking up ``in_indices[flat_map[i]]``
  for each output position.
  Used by handlers that already have a precomputed flat integer map
  (broadcast, tile, gather).
- **`_transform_indices(in_indices, in_shape, transform)`** —
  builds output index sets by applying ``transform`` to a position map of ``in_shape``.
  The transform function receives an ndarray and returns an ndarray;
  the result is raveled and passed to ``_permute_indices``.
  Used by handlers where each output reads exactly one input element
  (transpose, rev, slice, reshape, split, dynamic_slice).
- **`_propagate_const_unary(eqn, state, transform)`** —
  propagates a const value through a unary op by applying `transform`.
  Mirrors `_propagate_const_binary` for the single-input case.
- **`_broadcast_flat_map(in_shape, out_shape)`** —
  maps each output position to the input position it reads
  under numpy broadcasting rules (size-1 dims read index 0).
  The single broadcasting implementation,
  shared by `_binary_elementwise`, `_clear_where_zero`, and `broadcast_in_dim`.
- **`_bounded_ranges(bounds)`** —
  builds the per-element candidate ranges for ``_enumerate_bounded_patterns``
  from flattened ``(lo, hi)`` bounds.
- **`_enumerate_bounded_patterns(ranges, out_size, make_pattern)`** —
  enumerates all candidate index combinations from ``ranges``
  (capped at ``_MAX_ENUM_COMBINATIONS``),
  calls ``make_pattern`` for each,
  and unions the results element-wise.
- **`_merge_index_dependencies(result, index_sets)`** —
  unions the index operand's own index sets into every enumerated pattern.
  Every handler with bounded enumeration
  (gather, scatter, dynamic_slice, dynamic_update_slice)
  applies this to the ``_enumerate_bounded_patterns`` result.
- **`_report_issue(msg)`** —
  appends the standard report-an-issue request (with the GitHub URL)
  to an error message, keeping the phrasing in one place.
- **`_conservative_indices(all_indices, out_size)`** —
  conservative fallback where every output element depends on the union of all inputs.
- **`_atom_value_bounds(atom, state)`** —
  returns `(lo, hi)` bounds for an atom:
  exact `(val, val)` for constants, tracked bounds for bounded variables, or `None`.
- **`_binary_value_bounds(eqn, state)`** / **`_ternary_value_bounds(eqn, state)`** —
  return every operand's bounds for a binary or ternary op,
  or `None` if any of them is unknown.
  Each operand is checked before the next is read,
  so an input-dependent early operand does not force materializing
  a large later-operand const whose bounds would be discarded.
  Every bounds propagator with more than one operand goes through these,
  so the short-circuit order cannot drift between handlers.
- **`_forward_across_jaxpr_boundary(state, src_atoms, dst_vars)`** —
  transfers known const values and value bounds together
  across a nested-jaxpr boundary,
  so a call site cannot forward one and forget the other.
  Direction-neutral: callers pass outer invars to inner invars going in,
  and inner outvars to outer outvars coming back out.
  Consts are forwarded as stored, never materialized.

## Index Set Aliasing

Index sets in `state.indices` are **shared, not copied**.
Multiple output elements may reference the same `set[int]` object,
and output sets may alias input sets.
Handlers must therefore **never mutate** a set obtained from `state.indices` or `_index_sets()`.
Always build new sets (via `_union_all`, `|`, or the factory helpers) instead of mutating in place.

The same applies to whole **lists**:
pass-through handlers (squeeze, reshape, unary element-wise, convert_element_type, integer_pow)
alias the input list outright instead of copying it,
and `_conservative_indices` points every output position at one shared set.
Never mutate a list obtained from `state.indices` or `_index_sets()` either,
not even by replacing entries.
To change individual entries, build a new list first
(a shallow `list(...)` copy is enough, as in `_dynamic_update_for_starts` and `_clear_where_zero`).

Sharing is deliberate: copying costs O(nnz) per handler,
so alias sets and lists whenever the output is identical to the input by construction.
A handler that needs to accumulate via `|=` must own the target set,
either freshly built or explicitly copied first,
as `_fixed_point_loop` in `_while.py` does for the loop carries.

## Const Value Tracking

Handlers like `broadcast_in_dim`, `select_n`, and `propagate_const_elementwise`
propagate concrete values through `state.consts`.
This lets downstream handlers resolve static indices precisely.

**Invariant**: if a required const value is missing from `state.consts`,
the handler must assume the worst and return a conservative pattern.
This applies to `gather`, `scatter`, `dynamic_slice`, `dynamic_update_slice`,
`dot_general` (zero-skipping), and `mul` (zero-clearing).

## Value Bounds Tracking

`state.bounds` tracks per-element inclusive `(lo, hi)` integer bounds
for variables that are bounded but not statically constant
(e.g. the output of `argmax` over a small axis).

Bounds flow through three roles:
**producers** create bounds (`argmax`/`argmin`),
**propagators** forward them
(`add`, `sub`, `mul`, `div`, `integer_pow`, `max`, `min`, `clamp`,
`convert_element_type`, `broadcast_in_dim`, `select_n`),
and **consumers** use them to tighten sparsity
(`gather`, `scatter`, `dynamic_slice`, `dynamic_update_slice`, comparisons).

`max`, `min`, and `clamp` are what keep `jnp.clip` on an index from erasing its bounds.
`jnp.clip` lowers to `max` followed by `min`, not to `clamp`,
so all three need rules for the "clip an index into range" idiom to stay sparse.
Being monotone increasing in every operand,
all three are exact at the interval endpoints.

Propagators only fire when **every** operand is bounded.
An operand with no bounds could be anything,
and the `(lo, hi)` representation cannot express a half-open interval,
so the result is dropped rather than widened.

**Invariant**: if bounds are unavailable (`_atom_value_bounds` returns `None`),
the handler must assume the worst and return a conservative pattern.

## Zero-Sized Arrays

Handlers must handle zero-sized arrays (shapes containing a 0 dimension) gracefully.
If the output has zero elements, the handler should return an empty index set list `[]`.
Add an early return before any coordinate-mapping logic
(`np.ravel_multi_index`, `np.indices`, `np.reshape` into the array shape)
that would crash on zero-sized shapes.

## Adding a New Handler

1. Write `_prop_<name>(eqn, state)` in the appropriate module.
2. Add a `case` branch in `_prop_dispatch`.
3. Remove from the fallback `case` group if upgrading from conservative.
4. Add tests in the corresponding `tests/_interpret/test_<module>.py` file.

For primitives from external packages (Equinox, Flax, etc.),
place the handler in a dedicated subfolder (e.g., `_equinox/_select_if_vmap.py`)
with tests in `tests/_interpret/_equinox/`.

## Tests

Each handler module `_foo.py` has a corresponding test file `tests/_interpret/test_foo.py`.

## Writing Style

Use **semantic line breaks** everywhere:
one sentence or clause per line in docstrings, comments, and markdown.
This applies to all prose, not just docstrings.

Focus comments on **why**, not what.
Explain why a branch exists, why a particular approach was chosen, or why a fallback is needed.
Don't narrate what the code already says.

### Handler Docstring Style

1. **Semantic summary**: What the operation does and how dependencies flow.
2. **Math**: The Jacobian structure in concise mathematical notation.
3. **Example**: A concrete input/output trace showing dependency sets before and after.
4. **Jaxpr**: The `eqn.invars` and `eqn.params` layout the handler reads.
5. **URL**: Link to the JAX docs for the primitive, as a bare URL on the last line.

## References

- [Understanding jaxprs](https://docs.jax.dev/en/latest/jaxpr.html)
- [Writing custom jaxpr interpreters](https://docs.jax.dev/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html)
