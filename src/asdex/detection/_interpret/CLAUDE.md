# _interpret: Jaxpr Interpreter for Index Set Propagation

Propagates per-element dependency index sets through a jaxpr
to determine Jacobian sparsity patterns.

## Structure

- `__init__.py`: `_prop_jaxpr`, `_prop_dispatch`, and the conservative fallback.
- `_common.py`: shared types, `_PropState`, and helpers.
  Skim its docstrings before writing a handler, since most building blocks already exist there.
- `_foo.py` holds `_prop_foo` for the JAX primitive `foo`.
- Handlers for external packages (Equinox, Flax, etc.) live in subfolders such as `_equinox/`.

## Invariants

Every handler must uphold these.

1. **Never miss a nonzero.**
   A handler may report extra nonzeros,
   but a missing one silently corrupts the decompressed Jacobian.
   If a required const value or bound is missing, return a conservative pattern rather than guessing.
   Value-dependent handlers degrade in steps:
   an exact const first,
   then every candidate within known bounds, enumerated and unioned (capped by `_MAX_ENUM_COMBINATIONS`),
   and otherwise a conservative pattern.
2. **Patterns are global.**
   A pattern must hold for every input value,
   so value-dependent precision (zero-skipping, resolving gather/scatter indices)
   only uses values fixed at trace time: literals, closure constants, and what is computed from them.

## Design

- **Index sets track derivatives, not data flow.**
  An output depends on an input only if their partial derivative can be nonzero.
  Piecewise-constant ops (`floor`, `sign`, comparisons, `argmax`) therefore emit empty sets,
  and `custom_jvp_call` follows the JVP rule JAX differentiates, not its primal.
  `custom_vjp_call` still follows its primal, a known gap
  (see `test_custom_vjp_follows_primal_not_rule`).
- **Mirror the op on positions.**
  Structural handlers apply the numpy equivalent of the op to a position map
  (`_position_map`, `_transform_indices`, `_join_inputs`)
  instead of re-deriving index arithmetic,
  and match JAX semantics such as clamping out-of-bounds starts (`_clamp_starts`).
- **Nested jaxprs recurse through `_prop_jaxpr`**,
  which handlers receive as a `PropJaxprFn` argument to avoid circular imports.
  `cond` unions its branches since the taken branch is unknown,
  `while` iterates its carries to a fixed point since its trip count is unknown,
  and `scan` simulates its known trip count, stopping early once the carry repeats.
- **Fail loudly.**
  Unknown primitives raise `NotImplementedError`.
  The conservative fallback is an explicit opt-in group in `_prop_dispatch`.
  Handlers raise with `_report_issue` on jaxpr structure they don't understand,
  and `_prop_jaxpr` checks every handler's output count.
- **Pay only for what is read.**
  Index sets are aliased rather than copied (see Index Sets below),
  and consts are materialized to numpy only when a handler reads them,
  so helpers check operands in order and stop at the first unknown one.

## Propagation State

Every handler takes `(eqn, state)`.
`_PropState` bundles:

- `indices`: `Var` to `list[IndexSet]`, the dependency sets being tracked.
- `consts`: `Var` to statically known values, which let gather/scatter resolve indices precisely.
- `bounds`: `Var` to per-element inclusive integer `(lo, hi)` bounds
  for values that are bounded but not constant (e.g. `argmax` over a small axis).

Helpers that touch state take the whole bundle,
so adding a field never changes signatures.
`indices` is fresh per nested jaxpr, while every other field is shared across scopes.
JAX caches traced jaxprs, so value info must be overwritten on every entry into a jaxpr
(see the `_PropState` docstring).

## Const and Bounds Rules

- Bounds propagators fire only when every operand is bounded,
  since `(lo, hi)` cannot express a half-open interval.
- Write bounds only through `_set_value_bounds`,
  and widen operands with `_exact_ints` before arithmetic,
  so integer overflow drops the bounds instead of wrapping into a wrong interval.
- Read multi-operand bounds through `_binary_value_bounds` / `_ternary_value_bounds`.
- Cross nested-jaxpr boundaries with `_forward_across_jaxpr_boundary`,
  which moves consts and bounds together.

## Index Sets

`IndexSet` is `set[int]` behind a narrow seam,
so the backend can be swapped cheaply (see the `IndexSet` docstring).
Handlers may rely on these operations and nothing else:

- **Construct** with `_empty_index_set`, `_singleton_index_set`, `_empty_index_sets`, `_identity_index_sets`.
  Never `set()` or `{i}`.
- **Union** with `|`, `|=`, `_union_all`, or `_union_elementwise`.
- **Test emptiness** by truthiness.
- **Copy** with `_copy_index_set` / `_copy_index_sets`, never `.copy()`.
- **Annotate** as `IndexSet` / `list[IndexSet]`, never `set[int]`.

Only `_coo_from_index_sets` in `detection/_api.py` iterates a set.

**Aliasing**: sets and lists in `state.indices` are shared, not copied.
Outputs alias inputs, pass-through handlers alias whole lists,
and `_conservative_indices` reuses one set for every position.
Never mutate a set or list read from state.
To accumulate with `|=`, copy the target set first.
This beats collecting operands for `_union_all`, whose temporary list per element dominates for small sets.
To replace list entries, shallow-copy the list first.
Alias whenever the output equals the input by construction, since copying costs O(nnz).

## Conventions

- **Terms**: "index sets" means `list[IndexSet]`.
  "map" means an integer array mapping output positions to input positions.
  Avoid "deps".
- **Names**: `in_indices`, `in_shape`, `in_val` (or `in1_val` / `in2_val`, or role names like `lhs_val`, `pred_val`),
  `in_bounds` (or `in1_bounds` / `in2_bounds`), `flat_map`.
- **Zero-sized outputs**: return `[]` early,
  before `np.ravel_multi_index`, `np.indices`, or a reshape into the output shape.
- **Writing style**: semantic line breaks everywhere (one sentence or clause per line).
  Comments explain why, not what.

## Adding a Handler

1. Write `_prop_<name>(eqn, state)` in its module, or in a subfolder for external packages.
2. Add a `case` in `_prop_dispatch`, removing it from the fallback group if present.
3. Add tests in `tests/_interpret/test_<module>.py` (see `tests/CLAUDE.md`).
4. Write the docstring as: semantic summary, Jacobian structure in math,
   example trace of index sets, the `eqn.invars` / `eqn.params` layout read,
   and the bare JAX docs URL on the last line.

## References

- [Understanding jaxprs](https://docs.jax.dev/en/latest/jaxpr.html)
- [Writing custom jaxpr interpreters](https://docs.jax.dev/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html)
