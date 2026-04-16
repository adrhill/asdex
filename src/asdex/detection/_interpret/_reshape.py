"""Propagation rule for reshape operations."""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    StateConsts,
    StateIndices,
    atom_shape,
    index_sets,
    numel,
    propagate_const_unary,
    transform_indices,
)


def prop_reshape(
    eqn: JaxprEqn, state_indices: StateIndices, state_consts: StateConsts
) -> None:
    """Reshape changes array shape without changing data or element count.

    Dependencies pass through unchanged in row-major (C) order.
    The Jacobian is the identity matrix.

    When ``dimensions`` is not None, JAX transposes the input axes
    before reshaping (e.g. ``ravel(order='F')`` emits ``dimensions=(1, 0)``).
    The permutation reorders which flat input each flat output reads from.

    Example: reshape([a,b,c,d], (2,2)) → [[a,b],[c,d]]
        Input state_indices:  [{0}, {1}, {2}, {3}]
        Output state_indices: [{0}, {1}, {2}, {3}]

    Example: reshape([[a,b,c],[d,e,f]], (6,), dimensions=(1,0))
        Transpose first → [[a,d],[b,e],[c,f]], then flatten → [a,d,b,e,c,f]
        Input state_indices:  [{0}, {1}, {2}, {3}, {4}, {5}]
        Output state_indices: [{0}, {3}, {1}, {4}, {2}, {5}]

    Jaxpr:
        invars[0]: operand — array to reshape
        new_sizes: target shape
        dimensions: optional axis permutation applied before reshape

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.reshape.html
    """
    in_indices = index_sets(state_indices, eqn.invars[0])
    out_size = numel(atom_shape(eqn.outvars[0]))
    if len(in_indices) != out_size:
        msg = (
            f"Reshape size mismatch: input has {len(in_indices)} elements "
            f"but output expects {out_size}. "
            "Please help out asdex's development by reporting this at "
            "https://github.com/adrhill/asdex/issues"
        )
        raise ValueError(msg)

    dimensions = eqn.params.get("dimensions")
    if dimensions is not None:
        # dimensions is a permutation applied before the reshape.
        # Build the flat index mapping: position map transposed then raveled
        # tells us which original flat index each output position reads.
        in_shape = atom_shape(eqn.invars[0])
        state_indices[eqn.outvars[0]] = transform_indices(
            in_indices, in_shape, lambda p: p.transpose(dimensions)
        )
    else:
        state_indices[eqn.outvars[0]] = in_indices

    new_sizes = eqn.params["new_sizes"]

    def _reshape_val(v: np.ndarray) -> np.ndarray:
        if dimensions is not None:
            return v.transpose(dimensions).reshape(new_sizes)
        return v.reshape(new_sizes)

    propagate_const_unary(eqn, state_consts, _reshape_val)
