# Computing Sparse Jacobians

`asdex` computes sparse Jacobians for functions
\(f: \mathbb{R}^n \to \mathbb{R}^m\)
using [row or column coloring](../explanation/coloring.md) with forward- or reverse-mode AD.

!!! tip "Verify correctness at least once"

    asdex's [sparsity patterns](../explanation/global-sparsity.md) should always be conservative,
    but a bug in [sparsity detection](../explanation/sparsity-detection.md) could cause missing nonzeros.
    Always verify against vanilla JAX at least once on a new function.
    See [Verifying Results](#verifying-results) below.

## Basic Usage

Pass your function and its `input_shape` to [`jacobian`](../reference/index.md#asdex.jacobian):

```python
import jax
from asdex import jacobian

jac_fn = jax.jit(jacobian(f, input_shape=1000))
J = jac_fn(x)
```

This runs the computationally expensive sparsity detection and coloring steps when defining `jac_fn`.
Subsequent calls to `jac_fn` only need to perform the cheap decompression step.
The result is a JAX [BCOO](https://docs.jax.dev/en/latest/jax.experimental.sparse.html) sparse matrix.

The same function can be reused across evaluations at different inputs:

```python
for x in inputs:
    J = jac_fn(x)
```

`asdex` supports multi-dimensional input and output arrays.
The Jacobian is always returned as a 2D matrix
of shape \((m, n)\) where \(n\) is the total number of input elements
and \(m\) is the total number of output elements

## Getting the Primal Value Too

Use [`value_and_jacobian`](../reference/index.md#asdex.value_and_jacobian)
or [`value_and_jacobian_from_coloring`](../reference/index.md#asdex.value_and_jacobian_from_coloring)
to get `(f(x), J)` without a redundant forward pass.

## Precomputing the Colored Pattern

For more control,
precompute the coloring explicitly:

```python exec="true" session="jac-precompute" source="above"
from asdex import jacobian_coloring, jacobian_from_coloring

def f(x):
    return (x[1:] - x[:-1]) ** 2

coloring = jacobian_coloring(f, input_shape=100)
```

```python exec="true" session="jac-precompute"
print(f"```\n{coloring}\n```")
```

This is useful when you want to visually inspect the coloring for correctness,
or save it to disk to avoid recomputation.
Pass the coloring to [`jacobian_from_coloring`](../reference/index.md#asdex.jacobian_from_coloring) to compute the Jacobian:

```python
jac_fn = jax.jit(jacobian_from_coloring(f, coloring))

for x in inputs:
    J = jac_fn(x)
```

!!! tip

    If your `coloring` looks wrong or overly dense,
    please help out `asdex`'s development by
    [reporting it](https://github.com/adrhill/asdex/issues).
    These reports directly drive improvements
    and are one of the most impactful ways to contribute.

## Saving and Loading Patterns

Save a coloring to disk and reload it in a later session:

```python
from asdex import jacobian_coloring

coloring = jacobian_coloring(f, input_shape=1000)
coloring.save("colored.npz")
```

```python
from asdex import ColoredPattern, jacobian_from_coloring

coloring = ColoredPattern.load("colored.npz")
jac_fn = jax.jit(jacobian_from_coloring(f, coloring))
```

[`SparsityPattern`](../reference/index.md#asdex.SparsityPattern) supports the same `save`/`load` interface.

## Choosing Row vs Column Coloring

By default, `asdex` tries both row and column coloring
and picks whichever needs fewer colors:

```python exec="true" session="jac" source="above"
from asdex import jacobian_coloring

def f(x):
    return (x[1:] - x[:-1]) ** 2

# Automatic selection (default):
coloring = jacobian_coloring(f, input_shape=100)
```

```python exec="true" session="jac"
print(f"```\n{coloring}\n```")
```

You can also force a specific AD mode.
``"fwd"`` colors columns (uses JVPs, forward-mode AD),
``"rev"`` colors rows (uses VJPs, reverse-mode AD):

```python exec="true" session="jac" source="above"
# Force forward mode (column coloring, uses JVPs):
coloring = jacobian_coloring(f, input_shape=100, mode="fwd")

# Force reverse mode (row coloring, uses VJPs):
coloring = jacobian_coloring(f, input_shape=100, mode="rev")
```

```python exec="true" session="jac"
print(f"```\n{coloring}\n```")
```

The one-call [`jacobian`](../reference/index.md#asdex.jacobian) API accepts the same `mode` parameter:

```python
jac_fn = jax.jit(jacobian(f, input_shape=100, mode="rev"))
```

When the number of colors is equal,
`asdex` prefers column coloring since JVPs are generally cheaper to compute in JAX.

## Separate Detection and Coloring

For even more control, you can split detection and coloring:

```python
from asdex import jacobian_sparsity, jacobian_coloring_from_sparsity

sparsity = jacobian_sparsity(f, input_shape=1000)
coloring = jacobian_coloring_from_sparsity(sparsity, mode="fwd")
```

This is useful when you want to manually provide a sparsity pattern.

## Manually Providing a Sparsity Pattern

You can provide a sparsity pattern manually if you already know it ahead of time.
Create a `SparsityPattern` from coordinate arrays, a dense matrix, or a JAX BCOO matrix.

From a dense boolean or numeric matrix:

```python exec="true" session="jac" source="above"
import numpy as np
from asdex import SparsityPattern

dense = np.array([[1, 1, 0, 0],
                  [0, 1, 1, 0],
                  [0, 0, 1, 1]])
sparsity = SparsityPattern.from_dense(dense)
```
```python exec="true" session="jac"
print(f"```\n{sparsity}\n```")
```

From row and column index arrays:

```python exec="true" session="jac" source="above"
sparsity = SparsityPattern.from_coo(
    rows=[0, 0, 1, 1, 2, 2],
    cols=[0, 1, 1, 2, 2, 3],
    shape=(3, 4),
)
```
```python exec="true" session="jac"
print(f"```\n{sparsity}\n```")
```

From a JAX BCOO sparse matrix:
```python
sparsity = SparsityPattern.from_bcoo(bcoo_matrix)
```

Finally, color the sparsity pattern and compute the Jacobian:
```python
from asdex import jacobian_coloring_from_sparsity, jacobian_from_coloring

coloring = jacobian_coloring_from_sparsity(sparsity)
jac_fn = jax.jit(jacobian_from_coloring(f, coloring))
J = jac_fn(x)
```

## Verifying Results

Use [`check_jacobian_correctness`][asdex.check_jacobian_correctness]
to verify `asdex`'s sparse Jacobian against vanilla JAX.

```python
from asdex import check_jacobian_correctness, jacobian_coloring

coloring = jacobian_coloring(f, input_shape=x.shape)
check_jacobian_correctness(f, x, coloring)
```

Use verification for debugging and initial setup, not in production loops.
A good place to call it is in your test suite.

By default, this uses randomized matrix-vector products (`method="matvec"`)
to check the sparse Jacobian against JVPs or VJPs.
The AD mode is derived from the coloring.
This is cheap — O(k) in the number of probes — and scales to large problems.
If the results match, the function returns silently.
If they disagree, it raises a [`VerificationError`][asdex.VerificationError].

You can also set custom tolerances, the number of probes, and the PRNG seed:

```python
check_jacobian_correctness(f, x, coloring, rtol=1e-5, atol=1e-5, num_probes=50, seed=42)
```

For an exact element-wise comparison against the full dense Jacobian,
use `method="dense"`:

```python
check_jacobian_correctness(f, x, coloring, method="dense")
```

!!! warning "Dense computation"

    `method="dense"` materializes the full dense Jacobian,
    which is computationally very expensive for large problems.
