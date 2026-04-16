"""Jacobian and Hessian sparsity detection via jaxpr graph analysis."""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from asdex.detection._interpret import prop_jaxpr
from asdex.detection._interpret._commons import identity_index_sets
from asdex.pattern import SparsityPattern


def jacobian_sparsity(
    f: Callable, input_shape: int | tuple[int, ...]
) -> SparsityPattern:
    """Detect global Jacobian sparsity pattern for f: R^n -> R^m.

    Analyzes the computation graph structure directly,
    without evaluating any derivatives.
    The result is valid for all inputs.

    Args:
        f: Function taking an array and returning an array.
        input_shape: Shape of the input array.
            An integer is treated as a 1D length.

    Returns:
        SparsityPattern of shape ``(m, n)``
            where ``n = prod(input_shape)`` and ``m = prod(output_shape)``.
            Entry ``(i, j)`` is present if output ``i`` depends on input ``j``.
    """
    dummy_input = jnp.zeros(input_shape)
    closed_jaxpr = jax.make_jaxpr(f)(dummy_input)
    jaxpr = closed_jaxpr.jaxpr
    m = int(jax.eval_shape(f, dummy_input).size)
    n = input_shape if isinstance(input_shape, int) else math.prod(input_shape)

    # Initialize: input element i depends on input index i
    input_indices = [identity_index_sets(n)]

    # Build state_consts from closed jaxpr consts for static index tracking
    state_consts = {
        var: np.asarray(val)
        for var, val in zip(jaxpr.constvars, closed_jaxpr.consts, strict=False)
    }

    # Propagate through the jaxpr
    output_indices_list = prop_jaxpr(jaxpr, input_indices, state_consts)

    # Extract output dependencies (first output variable)
    out_indices = output_indices_list[0] if output_indices_list else []

    # Build sparsity pattern
    rows = []
    cols = []
    for i, deps in enumerate(out_indices):
        for j in deps:
            rows.append(i)
            cols.append(j)

    shape_tuple = (input_shape,) if isinstance(input_shape, int) else tuple(input_shape)
    return SparsityPattern.from_coo(rows, cols, (m, n), input_shape=shape_tuple)


def hessian_sparsity(
    f: Callable, input_shape: int | tuple[int, ...]
) -> SparsityPattern:
    """Detect global Hessian sparsity pattern for f: R^n -> R.

    Analyzes the Jacobian sparsity of the gradient function,
    without evaluating any derivatives.
    The result is valid for all inputs.

    If ``f`` returns a squeezable shape like ``(1,)`` or ``(1, 1)``,
    it is automatically squeezed to scalar.

    Args:
        f: Scalar-valued function taking an array.
        input_shape: Shape of the input array.
            An integer is treated as a 1D length.

    Returns:
        SparsityPattern of shape ``(n, n)``
            where ``n = prod(input_shape)``.
            Entry ``(i, j)`` is present if ``H[i, j]`` may be nonzero.
    """
    f = _ensure_scalar(f, input_shape)
    return jacobian_sparsity(jax.grad(f), input_shape)


def _ensure_scalar(f: Callable, input_shape: int | tuple[int, ...]) -> Callable:
    """Ensure ``f`` returns a scalar, auto-squeezing if possible.

    If ``f`` already returns shape ``()``, it is returned unchanged.
    If squeezing the output yields a scalar (e.g. shape ``(1,)``),
    a wrapped version is returned.
    Otherwise, raises ``ValueError``.
    """
    out = jax.eval_shape(f, jnp.zeros(input_shape))
    if out.shape == ():
        return f
    squeezed = jax.eval_shape(lambda x: jnp.squeeze(f(x)), jnp.zeros(input_shape))
    if squeezed.shape != ():
        raise ValueError(
            f"Expected scalar-valued function, but f has output shape {out.shape}."
        )
    return lambda x: jnp.squeeze(f(x))
