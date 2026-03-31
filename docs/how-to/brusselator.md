# Example: Brusselator PDE

This example computes the sparse Jacobian of a semi-discretized
2D reaction-diffusion system using `asdex`.

## The Brusselator System

The Brusselator models autocatalytic reactions
on a 2D domain \([0, 1]^2\) with periodic boundary conditions:

\[
\frac{\partial u}{\partial t} = 1 + u^2 v - 4.4\,u + \alpha \nabla^2 u
\]

\[
\frac{\partial v}{\partial t} = 3.4\,u - u^2 v + \alpha \nabla^2 v
\]

with diffusion coefficient \(\alpha = 10\).
See the [SciML Brusselator tutorials](#references) for the full formulation
including a localized forcing term,
which is omitted here since it is state-independent
and does not affect the Jacobian sparsity.

## Discretizing the RHS

Semi-discretize in space using second-order finite differences
on an \(N \times N\) grid.
The state vector concatenates both species \(u\) and \(v\),
giving \(2N^2\) unknowns:

```python exec="true" session="bruss" source="above"
import jax.numpy as jnp

N = 32
alpha = 10.0
dx = 1.0 / N

def brusselator_rhs(uv):
    u = uv[:N*N].reshape(N, N)
    v = uv[N*N:].reshape(N, N)

    # 5-point Laplacian with periodic boundary conditions
    def laplacian(w):
        return (
            jnp.roll(w, 1, axis=0) + jnp.roll(w, -1, axis=0)
            + jnp.roll(w, 1, axis=1) + jnp.roll(w, -1, axis=1)
            - 4 * w
        ) / dx**2

    du = 1.0 + u**2 * v - 4.4 * u + alpha * laplacian(u)
    dv = 3.4 * u - u**2 * v + alpha * laplacian(v)

    return jnp.concatenate([du.ravel(), dv.ravel()])
```

## Detecting and Coloring the Jacobian

The Jacobian has shape \(2048 \times 2048\),
but only 6 nonzeros per row
(5 from the Laplacian stencil plus 1 from reaction coupling).
Detect the sparsity and color it in one call:

```python exec="true" session="bruss" source="above"
from asdex import jacobian_coloring

coloring = jacobian_coloring(brusselator_rhs, input_shape=2 * N * N)
```

```python exec="true" session="bruss"
print(f"```\n{coloring}\n```")
```

Instead of 2048 JVPs or VJPs,
`asdex` needs only as many as there are colors.

## Computing the Jacobian

With the coloring precomputed,
evaluate the sparse Jacobian at any state:

```python exec="true" session="bruss" source="above"
from asdex import jacobian_from_coloring

# Brusselator initial condition
x = jnp.linspace(0, 1, N, endpoint=False)
xx, yy = jnp.meshgrid(x, x)
u0 = 22.0 * (yy * (1 - yy)) ** 1.5
v0 = 27.0 * (xx * (1 - xx)) ** 1.5
uv0 = jnp.concatenate([u0.ravel(), v0.ravel()])

jac_fn = jacobian_from_coloring(brusselator_rhs, coloring)
J = jac_fn(uv0)
```

```python exec="true" session="bruss"
print(f"```\n{J}\n```")
```

Make sure to reuse the `coloring` across evaluations at different states,
such that only the decompression step is repeated.

## References

This example is based on tutorials from the [SciML](https://sciml.ai) ecosystem.
Consider giving the [Julia programming language](https://julialang.org) a shot, it is fantastic.

- [NonlinearSolve.jl: _Efficiently Solving Large Sparse Ill-Conditioned Nonlinear Systems in Julia_](https://docs.sciml.ai/NonlinearSolve/stable/tutorials/large_systems/)
  (MIT License, [Copyright (c) 2020 Julia Computing, Inc.](https://github.com/SciML/NonlinearSolve.jl/blob/master/LICENSE))
- [MethodOfLines.jl: _Getting Started_](https://docs.sciml.ai/MethodOfLines/stable/tutorials/brusselator/)
  (MIT License, [Copyright (c) 2022 SciML Open Source Scientific Machine Learning Organization.](https://github.com/SciML/MethodOfLines.jl/blob/master/LICENSE))
