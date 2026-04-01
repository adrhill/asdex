# Getting Started

This tutorial walks through the three stages of [automatic sparse differentiation](../explanation/asd.md):
**[detection](../explanation/sparsity-detection.md)**, **[coloring](../explanation/coloring.md)**, and **decompression**.

## The Problem

For a function $f: \mathbb{R}^n \to \mathbb{R}^m$, computing the full Jacobian $J \in \mathbb{R}^{m \times n}$ requires $n$ forward-mode or $m$ reverse-mode AD passes.
In practice, for example in scientific machine learning,
many Jacobians are *sparse* (i.e., most entries are structurally zero, regardless of the input).

`asdex` exploits this sparsity in three steps:
1. **Detect** the sparsity pattern by tracing the computation graph
2. **Color** the pattern so that structurally orthogonal rows (or columns) share a color
3. **Decompress** one AD pass per color into the sparse Jacobian or Hessian

This reduces the computational cost from $m$ (or $n$) AD passes to just the number of colors,
yielding significant speedups on large sparse problems,
especially when the cost of detection and coloring can be amortized over repeated evaluations.
The same approach applies to sparse Hessians via forward-over-reverse AD.

## Sparse Jacobians

Consider the squared differences function
\(f(x)_i = (x_{i+1} - x_i)^2\),
which has a banded Jacobian.
Detect the sparsity pattern and color it in one step:

```python exec="true" session="gs" source="above"
import jax.numpy as jnp
from asdex import jacobian_coloring

def f(x):
    return (x[1:] - x[:-1]) ** 2

x = jnp.ones(50)

coloring = jacobian_coloring(f, input_shape=x.shape)
```

```python exec="true" session="gs"
print(f"```\n{coloring}\n```")
```

The print-out shows the original sparsity pattern (left) compressed into just two colors (right).
`asdex` automatically ran multiple coloring algorithms,
selected column coloring (2 JVPs) over row coloring (2 VJPs) since JVPs are cheaper,
reducing the cost from 49 VJPs or 50 JVPs without coloring to just 2 JVPs.
Note that on small problems, this doesn't directly translate into a speedup of factor 25x,
as the decompression overhead dominates.

!!! info "Global Sparsity Patterns"

    The detected pattern is a [global sparsity pattern](../explanation/global-sparsity.md):
    it depends only on the function's structure, not on any particular input.
    This means it may contain extra nonzeros compared to the sparsity at a specific point,
    but it is guaranteed to be correct everywhere and can therefore be reused.

    If you encounter overly conservative patterns,
    please [open an issue](https://github.com/adrhill/asdex/issues).
    These reports directly drive improvements
    and are one of the most impactful ways to contribute.


Now we can compute the sparse Jacobian using the coloring:

```python exec="true" session="gs" source="above"
import jax
from asdex import jacobian_from_coloring

jac_fn = jax.jit(jacobian_from_coloring(f, coloring))
J = jac_fn(x)
```

The result is a JAX
[BCOO](https://docs.jax.dev/en/latest/jax.experimental.sparse.html) sparse matrix.
We can verify that `asdex` produces the same result as `jax.jacobian`:

```python exec="true" session="gs" source="above"
import jax
import numpy as np

J_asdex = J.todense()
J_jax = jax.jacobian(f)(x)

np.testing.assert_allclose(J_asdex, J_jax, atol=1e-6)
```

`asdex` also provides [`check_jacobian_correctness`][asdex.check_jacobian_correctness]
as a convenience for this comparison —
see [Verifying Results](../how-to/jacobians.md#verifying-results).

On larger problems, the speedup from coloring becomes significant.
Let's benchmark on a 5000-dimensional input
(note that timings may vary as part of the doc-building process).
This time, we use [`asdex.jacobian`](../reference/index.md#asdex.jacobian), which calls [`jacobian_coloring`](../reference/index.md#asdex.jacobian_coloring) and [`jacobian_from_coloring`](../reference/index.md#asdex.jacobian_from_coloring):

```python exec="true" session="gs" source="above"
import asdex
import jax
import timeit

n = 5000
x = jnp.ones(n)

jac_fn_asdex = jax.jit(asdex.jacobian(f, input_shape=n))
jac_fn_jax = jax.jit(jax.jacobian(f))

# Warm up JIT caches
_ = jac_fn_asdex(x)
_ = jac_fn_jax(x)

t_asdex = timeit.timeit(lambda: jac_fn_asdex(x).block_until_ready(), number=10) / 10
t_jax = timeit.timeit(lambda: jac_fn_jax(x).block_until_ready(), number=10) / 10
```

```python exec="true" session="gs"
lines = [
    f"asdex.jacobian:  {t_asdex*1000:8.2f} ms",
    f"jax.jacobian:    {t_jax*1000:8.2f} ms",
    f"speedup:         {t_jax/t_asdex:8.1f}x",
]
print("```\n" + "\n".join(lines) + "\n```")
```

!!! tip "Precompute for Repeated Evaluations"

    The coloring depends only on the function structure,
    not the input values.
    When computing Jacobians at many different inputs,
    precompute the coloring once and reuse it:

    ```python
    jac_fn = jax.jit(jacobian(f, input_shape=5000))

    for x in inputs:
        J = jac_fn(x)
    ```

## Sparse Hessians

For scalar-valued functions \(f: \mathbb{R}^n \to \mathbb{R}\),
`asdex` can detect Hessian sparsity and compute sparse Hessians:

```python
import jax
from asdex import hessian

def g(x):
    return jnp.sum(x ** 2)

hess_fn = jax.jit(hessian(g, input_shape=20))

for x in inputs:
    H = hess_fn(x)
```

## Next Steps

- [Computing Sparse Jacobians](../how-to/jacobians.md) — Guide on Jacobian computation
- [Computing Sparse Hessians](../how-to/hessians.md) — Guide on Hessian computation
- [Sparsity Detection](../explanation/sparsity-detection.md) — Explanation how sparsity patterns are detected
- [Graph Coloring](../explanation/coloring.md) — Explanation how coloring reduces cost
