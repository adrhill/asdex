---
name: find-bugs
description: Find bugs in asdex by testing edge cases against JAX's reference implementations. Use when asked to hunt for bugs, test edge cases, or verify correctness.
allowed-tools: Bash(uv run python *) Bash(gh issue *)
---

# Find Bugs in asdex

Test asdex functions against JAX reference implementations to find correctness bugs.

## Workflow

1. Write test functions comparing `asdex.jacobian`/`asdex.hessian` against `jax.jacobian`/`jax.hessian`
2. Focus on edge cases: unusual primitives, PyTree structures, argnums combinations, control flow
3. When a test fails, minimize to an MWE
4. Check open issues: `gh issue list --state open`
5. File new issue with label "bug", title prefixed "Bug:", including description, MWE, pytest, and root cause

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

**Primitives** (check `src/asdex/detection/_interpret/__init__.py` `prop_dispatch`):
- Control flow: `cond`, `scan`, `while_loop`, `fori_loop`
- Index ops: `gather`, `scatter`, `dynamic_slice`, `dynamic_update_slice`
- Reductions: `reduce`, `reduce_window`, `reduce_max`, `reduce_min`
- Elementwise: `clamp`, `select`, `select_n`
