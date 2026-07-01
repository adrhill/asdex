---
name: find-bugs
description: Find bugs in asdex by testing edge cases against JAX's reference implementations. Use when asked to hunt for bugs, test edge cases, or verify correctness.
allowed-tools: Bash(uv run python *) Bash(gh issue *)
---

# Find Bugs in asdex

Test asdex functions against JAX reference implementations to find correctness bugs.

## Workflow

1. **First**, check open issues: `gh issue list --state open`
2. Write test functions comparing `asdex.jacobian`/`asdex.hessian` against `jax.jacobian`/`jax.hessian`
3. Focus on edge cases not covered by open issues
4. When a test fails or a `NotImplementedError` is raised:
   a. Skip if it matches an open issue (no need to re-confirm known bugs)
   b. For new bugs, minimize to an MWE and file an issue:
      - Bugs: label "bug", title prefixed "Bug:", include MWE and root cause
      - Missing primitives: label "enhancement", title prefixed "Feature:", include MWE
5. Continue testing other edge cases after filing

## Test Pattern

```python
import jax
import jax.numpy as jnp
import asdex

def f(x):
    return some_jax_operation(x)

x = jnp.array([...])
J_asdex = asdex.jacobian(f, x, output_format="dense")(x)
J_jax = jax.jacobian(f)(x)
assert jnp.allclose(J_asdex, J_jax)
```

## High-Value Test Areas

**Input/output combinations**:
- PyTree inputs: nested dicts, namedtuples, lists, mixed containers, empty leaves
- PyTree outputs: tuple, dict, nested structures, scalar leaves
- Multi-arg with argnums: reversed order `(1,0)`, non-contiguous `(0,2)`, negative `-1`
- Shape edge cases: scalars, empty arrays, 0-dim, high-dimensional tensors

**Save/load roundtrips**:
- `SparsityPattern` and `ColoredPattern` serialization
- Pickle roundtrips, JSON export/import if supported

**Primitives** (check `src/asdex/detection/_interpret/__init__.py` `_prop_dispatch`):
- Control flow: `cond`, `scan`, `while_loop`, `fori_loop`
- Index ops: `gather`, `scatter`, `dynamic_slice`, `dynamic_update_slice`
- Reductions: `reduce`, `reduce_window`, `reduce_max`, `reduce_min`
- Reduction edge cases: empty axis `axis=()`, `keepdims=True`, partial axis subsets
- Elementwise: `clamp`, `select`, `select_n`

## Sparsity Precision Testing

Check if asdex detects exact sparsity (not overly conservative):

```python
import numpy as np

pattern = asdex.jacobian_sparsity(f, x)
detected_nnz = pattern.nnz

J_jax = jax.jacobian(f)(x)
true_nnz = np.sum(np.abs(np.asarray(J_jax)) > 1e-10)

if detected_nnz > true_nnz:
    print(f"Conservative pattern: detected {detected_nnz}, true {true_nnz}")
```

Conservative patterns are correct but suboptimal. Document these with a `TODO(primitive)` comment showing the precise expected pattern.
Note that asdex computes global sparisty patterns that must be valid over the entire input domain: https://adrianhill.de/asdex/explanation/global-sparsity/
