# _interpret — Custom Jaxpr Interpreter for Index Set Propagation

Implements a custom jaxpr interpreter that propagates per-element dependency index sets (`set[int]`)
through primitives to determine Jacobian sparsity patterns.

## Structure

- `__init__.py` — `prop_jaxpr`, `prop_dispatch`, fallback handling.
- `_commons.py` — shared types (`IndexSet`, `StateIndices`, `StateConsts`) and utilities.
- Each JAX primitive has its own module: `_foo.py` contains `prop_foo`.
  Includes `_cumsum.py` for cumulative sum.
- Handlers for external packages (Equinox, Flax, etc.) live in their own subfolders
  (e.g., `_equinox/`).

## Key Types

- `IndexSet` = `set[int]` — a single per-element dependency set
- `list[IndexSet]` — per-element dependency sets for one array
- `StateIndices` = `dict[Var, list[IndexSet]]` — maps jaxpr variables to their index sets
- `StateConsts` = `dict[Var, np.ndarray]` — statically-known values for precise gather/scatter
- `StateBounds` = `dict[Var, tuple[np.ndarray, np.ndarray]]` — per-element inclusive (lo, hi) integer bounds

## Naming Conventions

**Terminology** — "indices" and "map" mean different things:
- **"indices" / "index sets"**: `list[IndexSet]`,
  the per-element dependency sets used for sparsity tracking.
- **"map"**: numpy integer arrays that map output positions to input positions.
  Not index sets.

**Construction** — always use the factory helpers from `_commons`:
- `empty_index_set()` instead of `set()`
- `singleton_index_set(i)` instead of `{i}`
- `empty_index_sets(n)` instead of `[set() for _ in range(n)]`
- `identity_index_sets(n)` instead of `[{i} for i in range(n)]`

This ensures a future backend swap only requires changing the helpers,
not every handler.

**Variable names** — use these consistently across handlers:
- `in_indices`: input index sets (from `index_sets(state_indices, atom)`)
- `in_shape`: input array shape (from `atom_shape(atom)`)
- `in_val`: const value for a unary input (from `atom_const_val(atom, state_consts)`)
- `in1_val` / `in2_val`: const values for binary inputs.
  Use descriptive prefixes when roles differ:
  `lhs_val` / `rhs_val` (dot_general), `pred_val` / `which_val` (select), etc.
- `in_bounds` / `in1_bounds` / `in2_bounds`: value bounds for inputs
  (from `atom_value_bounds(atom, state_consts, state_bounds)`)
- `flat_map`: a flat integer array mapping output positions to input positions

**Docstrings** — avoid the term "deps"; prefer "index sets" or "input index sets".

## Common Utilities in `_commons.py`

- **`position_map(shape)`** —
  builds an array where each element holds its own flat position.
  Applying operations (transpose, slice, flip) to this array
  reveals which input position each output position reads from.
- **`permute_indices(in_indices, flat_map)`** —
  builds output index sets by looking up ``in_indices[flat_map[i]]``
  for each output position.
  Used by handlers that already have a precomputed flat integer map
  (broadcast, tile, gather).
- **`transform_indices(in_indices, in_shape, transform)`** —
  builds output index sets by applying ``transform`` to a position map of ``in_shape``.
  The transform function receives an ndarray and returns an ndarray;
  the result is raveled and passed to ``permute_indices``.
  Used by handlers where each output reads exactly one input element
  (transpose, rev, slice, reshape, split, dynamic_slice).
- **`propagate_const_unary(eqn, state_consts, transform)`** —
  propagates a const value through a unary op by applying `transform`.
  Mirrors `propagate_const_binary` for the single-input case.
- **`enumerate_bounded_patterns(ranges, out_size, make_pattern)`** —
  enumerates all candidate index combinations from ``ranges``
  (capped at ``_MAX_ENUM_COMBINATIONS``),
  calls ``make_pattern`` for each,
  and unions the results element-wise.
- **`conservative_indices(all_indices, out_size)`** —
  conservative fallback where every output element depends on the union of all inputs.
- **`atom_value_bounds(atom, state_consts, state_bounds)`** —
  returns `(lo, hi)` bounds for an atom:
  exact `(val, val)` for constants, tracked bounds for bounded variables, or `None`.
- **`forward_value_bounds(state_bounds, outer_atoms, inner_vars)`** —
  transfers known value bounds from outer-scope atoms to inner jaxpr variables.

## Index Set Aliasing

Index sets in `StateIndices` are **shared, not copied**.
Multiple output elements may reference the same `set[int]` object,
and output sets may alias input sets.
Handlers must therefore **never mutate** a set obtained from `state_indices` or `index_sets()`.
Always build new sets (via `union_all`, `|`, or the factory helpers) instead of mutating in place.

The one exception is `_fixed_point_loop` in `_while.py`,
which explicitly copies carry sets before mutating them via `|=`.

## Const Value Tracking

Handlers like `broadcast_in_dim`, `select_n`, and `propagate_const_elementwise`
propagate concrete values through `state_consts`.
This lets downstream handlers resolve static indices precisely.

**Invariant**: if a required const value is missing from `state_consts`,
the handler must assume the worst and return a conservative pattern.
This applies to `gather`, `scatter`, `dynamic_slice`, `dynamic_update_slice`,
`dot_general` (zero-skipping), and `mul` (zero-clearing).

## Value Bounds Tracking

`StateBounds` tracks per-element inclusive `(lo, hi)` integer bounds
for variables that are bounded but not statically constant
(e.g. the output of `argmax` over a small axis).

Bounds flow through three roles:
**producers** create bounds (`argmax`/`argmin`),
**propagators** forward them (`add`, `sub`, `convert_element_type`, `broadcast_in_dim`, `select_n`),
and **consumers** use them to tighten sparsity
(`gather`, `scatter`, `dynamic_slice`, `dynamic_update_slice`, comparisons).

**Invariant**: if bounds are unavailable (`atom_value_bounds` returns `None`),
the handler must assume the worst and return a conservative pattern.

## Zero-Sized Arrays

Handlers must handle zero-sized arrays (shapes containing a 0 dimension) gracefully.
If the output has zero elements, the handler should return an empty index set list `[]`.
Add an early return before any coordinate-mapping logic
(`np.ravel_multi_index`, `np.indices`, `np.reshape` into the array shape)
that would crash on zero-sized shapes.

## Adding a New Handler

1. Write `prop_<name>(eqn, state_indices, ...)` in the appropriate module.
2. Add a `case` branch in `prop_dispatch`.
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
