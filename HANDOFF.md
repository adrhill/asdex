# Handoff: BCOO PyTree Structure Preservation

## Problem Summary

BCOO output format loses PyTree structure in various cases while dense format works correctly. There are 28 xfailing tests, 27 of which are BCOO-related.

## What Was Fixed

The dense format had two bugs in `_assemble_jacobian` (decompression.py:1026-1035) where early-return optimizations incorrectly assumed "single leaf = trivial structure":

1. **Single output leaf**: Wrapped in input structure but forgot output structure
2. **Single input leaf**: Wrapped in output structure but forgot input structure

**Fix**: Removed both early returns. Now always uses the general path:
```python
out_trees_per_in_leaf = [
    jax.tree_util.tree_unflatten(out_treedef, out_blocks)
    for out_blocks in per_input_blocks
]
in_tree_of_out_trees = _group_blocks_by_argnums(out_trees_per_in_leaf, sparsity)
return _transpose_in_out_trees(in_tree_of_out_trees, out_treedef, output_format)
```

This follows JAX's pattern of always building both tree structures and transposing.

## What Remains: BCOO Issues

### Root Cause

The issue is in `_transpose_in_out_trees` (decompression.py:1100-1146). For BCOO format, it uses a manual transpose to avoid `tree_transpose` seeing BCOO's internal pytree structure:

```python
def _transpose_in_out_trees(in_tree_of_out_trees, out_treedef, output_format):
    # ...
    if output_format == "dense":
        return jax.tree_util.tree_transpose(in_treedef, out_treedef, in_tree_of_out_trees)

    # BCOO: manual transpose to avoid tree_transpose seeing BCOO's internal structure.
    out_trees = jax.tree_util.tree_leaves(in_tree_of_out_trees, is_leaf=is_out_tree)
    leaves_per_out_tree = [
        jax.tree_util.tree_leaves(t, is_leaf=is_bcoo) for t in out_trees
    ]
    # ... manual transpose logic ...
```

The manual BCOO path has bugs:
1. `is_out_tree` detection fails for deeply nested structures
2. The manual transpose doesn't preserve all levels of nesting
3. Single-leaf pytrees get collapsed to just the leaf

### Failing Test Categories

**Single-leaf dict** (8 tests): `{"w": array}` becomes just `array`
```python
# Expected: {"w": jacobian_block}
# Actual:   jacobian_block
```

**Nested dict inputs** (5 tests): `{"layer": {"w": array}}` loses structure
```python
# Expected: {"layer": {"w": jacobian_block}}
# Actual:   jacobian_block or {"w": jacobian_block}
```

**argnums tuple vs int** (3 tests): `argnums=(0,)` should return `(J,)` not `J`
```python
# Expected: (jacobian,)  # tuple with one element
# Actual:   jacobian     # unwrapped
```

**Single PyTree position** (2 tests): `argnums=0` with pytree input loses structure

### Suggested Fix

**Option A: Fix the manual BCOO transpose**

The `is_out_tree` check uses `tree_structure(x) == out_treedef`, but this fails when:
- The structure has nested containers with single leaves
- BCOO arrays are at different nesting levels

Fix by tracking the expected structure more carefully:

```python
def _transpose_in_out_trees(in_tree_of_out_trees, out_treedef, output_format):
    def is_bcoo(x):
        return isinstance(x, BCOO)

    # For BCOO, wrap each BCOO in a marker, transpose, then unwrap
    if output_format == "bcoo":
        # Mark BCOO arrays so tree_transpose doesn't see their internals
        @dataclass
        class BCOOMarker:
            value: BCOO

        register_pytree_node(BCOOMarker, lambda m: ((m.value,), None), lambda _, xs: BCOOMarker(xs[0]))

        marked = jax.tree_util.tree_map(
            lambda x: BCOOMarker(x) if isinstance(x, BCOO) else x,
            in_tree_of_out_trees,
            is_leaf=is_bcoo
        )
        transposed = jax.tree_util.tree_transpose(in_treedef, out_treedef, marked)
        return jax.tree_util.tree_map(
            lambda x: x.value if isinstance(x, BCOOMarker) else x,
            transposed,
            is_leaf=lambda x: isinstance(x, BCOOMarker)
        )

    return jax.tree_util.tree_transpose(in_treedef, out_treedef, in_tree_of_out_trees)
```

**Option B: Use a simpler BCOO wrapper**

JAX's `tree_transpose` works correctly if BCOO isn't registered as a pytree. Consider wrapping BCOO in a simple non-pytree container during transpose:

```python
class _BCOOLeaf:
    """Wrapper to hide BCOO's internal pytree structure from tree operations."""
    __slots__ = ('array',)
    def __init__(self, array): self.array = array
```

Then wrap before transpose, transpose normally, unwrap after.

**Option C: Convert BCOO to/from COO indices**

Before transpose, extract BCOO data/indices as plain arrays. After transpose, reconstruct BCOO. This avoids pytree issues entirely but may be slower.

## Non-BCOO Issue

`test_jacobian_allow_int_pytree_input` fails because `float0` dtype (returned for integer inputs with `allow_int=True`) can't be concatenated with regular floats in `_flatten_selected_cotangents`:

```python
def _flatten_selected_cotangents(cotangents, sparsity):
    # ...
    return jnp.concatenate([leaf.ravel() for leaf in leaves])  # Fails if mixed float0/float
```

**Fix**: Filter out or specially handle `float0` leaves before concatenation.

## Test Commands

```bash
# Run all e2e tests
uv run pytest tests/e2e/ -v

# Run only the xfailing BCOO tests
uv run pytest tests/e2e/ -v -k "bcoo" --runxfail

# Run specific failing test to debug
uv run pytest tests/e2e/test_pytree.py::test_jacobian_single_leaf_pytree -v --runxfail
```

## Files to Modify

- `src/asdex/decompression.py`: `_transpose_in_out_trees` (lines 1100-1146)
- `src/asdex/decompression.py`: `_flatten_selected_cotangents` (line 914, for `allow_int` fix)
- `tests/e2e/*.py`: Remove xfails as bugs are fixed
