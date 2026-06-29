# Verifying Correctness

asdex's [sparsity patterns](../explanation/global-sparsity.md) should always be conservative,
but a bug in [sparsity detection](../explanation/sparsity-detection.md) could drop a nonzero,
resulting in wrong Jacobians or Hessians.
Verify asdex' results against vanilla JAX at least once on every new function.
This guide shows you how.

## Jacobians

[`check_jacobian_correctness`][asdex.check_jacobian_correctness] compares asdex's sparse Jacobian against a JAX reference.
It returns silently on success and raises a [`VerificationError`][asdex.VerificationError] on mismatch.

```python exec="true" session="verify" source="above"
import jax.numpy as jnp
from asdex import jacobian_coloring, check_jacobian_correctness

def f(x):
    return (x[1:] - x[:-1]) ** 2

x = jnp.arange(1.0, 11.0)
coloring = jacobian_coloring(f, x)
check_jacobian_correctness(f, x, coloring)  # no output means it matches JAX
```

The call above produces no output: success is silent, and a mismatch would raise instead.
By default this uses `method="matvec"`, computing randomized matrix-vector products (i.e., JVPs, VJPs, or HVPs, depending on the coloring).
This is cheap, \(O(k)\) in the number of probes, and scalable.
You can tune the probes, tolerances, and seed:

```python exec="true" session="verify" source="above"
check_jacobian_correctness(f, x, coloring, num_probes=20, rtol=1e-5, atol=1e-5, seed=42)
```

For an exact but expensive element-wise comparison against the full dense Jacobian,
use `method="dense"`:

```python exec="true" session="verify" source="above"
check_jacobian_correctness(f, x, coloring, method="dense")
```

!!! warning "Dense comparison is expensive"

    `method="dense"` materializes the full dense Jacobian (\(O(n^2)\)),
    so reserve it for small problems.

## Hessians

[`check_hessian_correctness`][asdex.check_hessian_correctness] mirrors the Jacobian API:

```python exec="true" session="verify" source="above"
from asdex import hessian_coloring, check_hessian_correctness

def g(x):
    return jnp.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2)

coloring = hessian_coloring(g, x)
check_hessian_correctness(g, x, coloring)              # matvec (default)
check_hessian_correctness(g, x, coloring, method="dense")  # exact, expensive
```

## Validating a coloring directly

To check a coloring without evaluating derivatives,
use the coloring validators,
which raise [`InvalidColoringError`][asdex.InvalidColoringError] on a bad assignment:
[`check_coloring_rows`][asdex.check_coloring_rows] (reverse mode),
[`check_coloring_cols`][asdex.check_coloring_cols] (forward mode), and
[`check_coloring_symmetric`][asdex.check_coloring_symmetric] (Hessians).
