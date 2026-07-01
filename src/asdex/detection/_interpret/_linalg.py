"""Handlers for linear algebra primitives."""

import numpy as np
from jax._src.core import JaxprEqn

from asdex.detection._interpret._common import (
    IndexSet,
    StateIndices,
    _atom_shape,
    _empty_index_set,
    _index_sets,
    _union_all,
)


def _prop_qr(eqn: JaxprEqn, state_indices: StateIndices) -> None:
    """QR decomposition: A = QR where Q is orthogonal and R is upper triangular.

    Q depends on all inputs (conservative).
    R is upper triangular: lower triangle elements are always zero (no dependencies),
    upper triangle elements depend on all inputs.

    Jaxpr:
        invars: [A]
        outvars: [Q, R]
        params: full_matrices (bool), pivoting (bool), use_magma (optional)

    https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.qr.html
    """
    (invar,) = eqn.invars
    q_var, r_var = eqn.outvars

    # Collect all input index sets
    in_indices = _index_sets(state_indices, invar)
    combined = _union_all(in_indices)

    # Q: all elements depend on all inputs
    q_shape = _atom_shape(q_var)
    q_numel = int(np.prod(q_shape))
    state_indices[q_var] = [combined] * q_numel

    # R: upper triangular
    # Upper triangle (including diagonal) depends on all inputs
    # Lower triangle is always 0 (no dependencies)
    r_shape = _atom_shape(r_var)
    r_numel = int(np.prod(r_shape))

    if r_numel == 0:
        state_indices[r_var] = []
        return

    nrows, ncols = r_shape[-2], r_shape[-1]
    batch_shape = r_shape[:-2]
    batch_size = int(np.prod(batch_shape)) if batch_shape else 1

    r_indices: list[IndexSet] = []
    for _ in range(batch_size):
        for i in range(nrows):
            for j in range(ncols):
                if i <= j:
                    # Upper triangle (including diagonal): depends on all
                    r_indices.append(combined)
                else:
                    # Lower triangle: always zero
                    r_indices.append(_empty_index_set())

    state_indices[r_var] = r_indices
