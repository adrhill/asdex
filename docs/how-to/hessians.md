# Computing Sparse Hessians

`asdex` computes sparse Hessians for scalar-valued functions
\(f: \mathbb{R}^n \to \mathbb{R}\)
using symmetric coloring and forward-over-reverse AD.

!!! tip "Verify correctness at least once"

    asdex's [sparsity patterns](../explanation/global-sparsity.md) should always be conservative,
    but a bug in [sparsity detection](../explanation/sparsity-detection.md) could cause missing nonzeros.
    Always verify against vanilla JAX at least once on a new function.
    See [Verifying Results](#verifying-results) below.

## Basic Usage

Pass your scalar-valued function and a sample input to [`hessian`](../reference/index.md#asdex.hessian):

```python
import jax
import jax.numpy as jnp
from asdex import hessian

x_sample = jnp.zeros(100)  # sample input for sparsity detection
hess_fn = jax.jit(hessian(f, x_sample))
H = hess_fn(x)
```

This runs the computationally expensive sparsity detection and coloring steps when defining `hess_fn`.
Subsequent calls to `hess_fn` only need to perform the cheap decompression step.
The result is a JAX [BCOO](https://docs.jax.dev/en/latest/jax.experimental.sparse.html) sparse matrix.

The same function can be reused across evaluations at different inputs:

```python
for x in inputs:
    H = hess_fn(x)
```

`asdex` supports multi-dimensional input arrays.
The Hessian is always returned as a 2D matrix
of shape \((n, n)\) where \(n\) is the total number of input elements.

## Getting the Primal Value Too

Use [`value_and_hessian`](../reference/index.md#asdex.value_and_hessian)
or [`value_and_hessian_from_coloring`](../reference/index.md#asdex.value_and_hessian_from_coloring)
to get `(f(x), H)` without a redundant forward pass.

## Precomputing the Colored Pattern

For more control,
precompute the coloring explicitly:

```python exec="true" session="hess-precompute" source="above"
import jax.numpy as jnp
from asdex import hessian_coloring, hessian_from_coloring

def g(x):
    return jnp.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2)

x = jnp.zeros(100)
coloring = hessian_coloring(g, x)
```

```python exec="true" session="hess-precompute"
print(f"```\n{coloring}\n```")
```

This is useful when you want to visually inspect the coloring for correctness,
or save it to disk to avoid recomputation.
Pass the coloring to [`hessian_from_coloring`](../reference/index.md#asdex.hessian_from_coloring) to compute the Hessian:

```python
hess_fn = jax.jit(hessian_from_coloring(g, coloring))

for x in inputs:
    H = hess_fn(x)
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
import jax.numpy as jnp
from asdex import hessian_coloring

x = jnp.zeros(100)
coloring = hessian_coloring(g, x)
coloring.save("colored.npz")
```

```python
from asdex import ColoredPattern, hessian_from_coloring

coloring = ColoredPattern.load("colored.npz")
hess_fn = jax.jit(hessian_from_coloring(g, coloring))
```

[`SparsityPattern`](../reference/index.md#asdex.SparsityPattern) supports the same `save`/`load` interface.

## Symmetric Coloring

Hessians are symmetric (\(H = H^\top\)),
and `asdex` exploits this with *star coloring*
(Gebremedhin et al., 2005).
Symmetric coloring typically needs fewer colors than row or column coloring,
since both \(H_{ij}\) and \(H_{ji}\) can be recovered from a single coloring.

The convenience functions [`hessian_coloring`](../reference/index.md#asdex.hessian_coloring) and [`hessian`](../reference/index.md#asdex.hessian) use symmetric coloring automatically.
Here we use the [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function),
a classic optimization benchmark whose Hessian is tridiagonal:

\[f(x) = \sum_{i=1}^{n-1} \left[(1 - x_i)^2 + 100\,(x_{i+1} - x_i^2)^2\right]\]

```python exec="true" session="hess" source="above"
import jax.numpy as jnp
from asdex import hessian_coloring

def rosenbrock(x):
    return jnp.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2)

x = jnp.zeros(100)
coloring = hessian_coloring(rosenbrock, x)
```

```python exec="true" session="hess"
print(f"```\n{coloring}\n```")
```

## Separate Detection and Coloring

For even more control, you can split detection and coloring:

```python
import jax.numpy as jnp
from asdex import hessian_sparsity, hessian_coloring_from_sparsity

x = jnp.zeros(100)
sparsity = hessian_sparsity(g, x)
coloring = hessian_coloring_from_sparsity(sparsity)
```

Since the Hessian is the Jacobian of the gradient,
`hessian_sparsity` simply calls `jacobian_sparsity(jax.grad(f), x)`.
The [sparsity interpreter](../explanation/sparsity-detection.md) composes naturally with JAX's autodiff transforms.

This is useful when you want to manually provide a sparsity pattern.

## Manually Providing a Sparsity Pattern

You can provide a sparsity pattern manually if you already know it ahead of time.
Create a `SparsityPattern` from coordinate arrays, a dense matrix, or a JAX BCOO matrix.

From a dense boolean or numeric matrix:

```python exec="true" session="hess" source="above"
import numpy as np
from asdex import SparsityPattern

dense = np.array([[1, 1, 0, 0],
                  [1, 1, 1, 0],
                  [0, 1, 1, 1],
                  [0, 0, 1, 1]])
sparsity = SparsityPattern.from_dense(dense)
```
```python exec="true" session="hess"
print(f"```\n{sparsity}\n```")
```

From row and column index arrays:

```python exec="true" session="hess" source="above"
sparsity = SparsityPattern.from_coo(
    rows=[0, 0, 1, 1, 1, 2, 2, 2, 3, 3],
    cols=[0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
    shape=(4, 4),
)
```
```python exec="true" session="hess"
print(f"```\n{sparsity}\n```")
```

From a JAX BCOO sparse matrix:
```python
sparsity = SparsityPattern.from_bcoo(bcoo_matrix)
```

Finally, color the sparsity pattern and compute the Hessian:
```python
from asdex import hessian_coloring_from_sparsity, hessian_from_coloring

coloring = hessian_coloring_from_sparsity(sparsity)
hess_fn = jax.jit(hessian_from_coloring(f, coloring))
H = hess_fn(x)
```

## Choosing an HVP Mode

By default, `hessian` uses forward-over-reverse AD to compute Hessian-vector products.
You can select a different AD composition strategy via the `mode` parameter:

```python
import jax.numpy as jnp
from asdex import hessian

x = jnp.zeros(100)
hess_fn_for = jax.jit(hessian(f, x, mode="fwd_over_rev"))  # default
hess_fn_rof = jax.jit(hessian(f, x, mode="rev_over_fwd"))
hess_fn_ror = jax.jit(hessian(f, x, mode="rev_over_rev"))
```

All three modes produce the same mathematical result.
They differ in how JAX's AD primitives are composed:

- **`fwd_over_rev`** (default): `jvp(grad(f), ...)`.
    Generally the fastest under JIT.
- **`rev_over_fwd`**: `grad(lambda p: jvp(f, (p,), (v,))[1])`.
    Can use less memory than forward-over-reverse for functions with many intermediates.
- **`rev_over_rev`**: `grad(lambda y: dot(grad(f)(y), v))`.
    Avoids forward-mode entirely;
    useful when forward-mode is expensive or unsupported.

!!! tip

    When in doubt, stick with the default `"fwd_over_rev"`.
    It is the most widely used and typically the most efficient under `jax.jit`.

## Verifying Results

Use [`check_hessian_correctness`][asdex.check_hessian_correctness]
to verify `asdex`'s sparse Hessian against vanilla JAX.

```python
from asdex import check_hessian_correctness, hessian_coloring

coloring = hessian_coloring(g, x)
check_hessian_correctness(g, x, coloring)
```

Use verification for debugging and initial setup, not in production loops.
A good place to call it is in your test suite.

By default, this uses randomized matrix-vector products (`method="matvec"`)
to check the sparse Hessian against an HVP reference.
The AD mode is derived from the coloring.
This is cheap — O(k) in the number of probes — and scales to large problems.
If the results match, the function returns silently.
If they disagree, it raises a [`VerificationError`][asdex.VerificationError].

You can also set custom tolerances, the number of probes, and the PRNG seed:

```python
check_hessian_correctness(g, x, coloring, rtol=1e-5, atol=1e-5, num_probes=50, seed=42)
```

For an exact element-wise comparison against the full dense Hessian,
use `method="dense"`:

```python
check_hessian_correctness(g, x, coloring, method="dense")
```

!!! warning "Dense computation"

    `method="dense"` materializes the full dense Hessian,
    which is computationally very expensive for large problems.
