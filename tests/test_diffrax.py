"""Integration tests for diffrax tracing."""

import diffrax
import jax.numpy as jnp
import numpy as np
import pytest

import asdex

N = 4  # smaller grid for faster tests
A = 1.0
B = 1.0
ALPHA = 1.0
DX = 1.0


def brusselator_rhs(y: jnp.ndarray) -> jnp.ndarray:
    """Brusselator RHS with periodic BCs via 5-point Laplacian."""
    n2 = N * N
    u = y[:n2].reshape(N, N)
    v = y[n2:].reshape(N, N)

    lap_u = (
        jnp.roll(u, 1, axis=0)
        + jnp.roll(u, -1, axis=0)
        + jnp.roll(u, 1, axis=1)
        + jnp.roll(u, -1, axis=1)
        - 4 * u
    ) / (DX * DX)

    lap_v = (
        jnp.roll(v, 1, axis=0)
        + jnp.roll(v, -1, axis=0)
        + jnp.roll(v, 1, axis=1)
        + jnp.roll(v, -1, axis=1)
        - 4 * v
    ) / (DX * DX)

    du = ALPHA * lap_u + B + u**2 * v - (A + 1) * u
    dv = ALPHA * lap_v + A * u - u**2 * v

    return jnp.concatenate([du.ravel(), dv.ravel()])


solver = diffrax.Euler()
term = diffrax.ODETerm(lambda t, y, args: brusselator_rhs(y))
stepsize = diffrax.ConstantStepSize()


def euler_step(y0: jnp.ndarray) -> jnp.ndarray:
    """One Euler step from t=0 to t=0.1."""
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=0.1,
        dt0=0.1,
        y0=y0,
        stepsize_controller=stepsize,
        max_steps=1,
    )
    return sol.ys[0]


@pytest.mark.control_flow
def test_diffrax_euler_step_traces():
    """Diffrax Euler step traces end-to-end without errors.

    The step pattern is a valid (conservative) superset of the RHS pattern.
    It is overestimated because the while loop fixed-point analysis
    iterates the body to convergence,
    which spreads dependencies through the Laplacian stencil.
    """
    n = 2 * N * N

    rhs_pattern = asdex.jacobian_sparsity(brusselator_rhs, input_shape=(n,))
    step_pattern = asdex.jacobian_sparsity(euler_step, input_shape=(n,))

    rhs_dense = rhs_pattern.todense().astype(bool)
    step_dense = step_pattern.todense().astype(bool)

    # The step pattern must be a superset of the RHS pattern.
    assert np.all(step_dense | ~rhs_dense)
    # The diagonal must be present (identity from y + dt*f(y)).
    assert np.all(np.diag(step_dense))
