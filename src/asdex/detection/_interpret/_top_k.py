"""Propagation rule for top_k.

The values output has the same sparsity structure as a reduction along the last axis.
The indices output has zero derivative.
"""

import math

from jax._src.core import JaxprEqn

from ._commons import (
    StateIndices,
    atom_numel,
    atom_shape,
    index_sets,
    union_all,
)


def prop_top_k(eqn: JaxprEqn, state_indices: StateIndices) -> None:
    """top_k selects the k largest elements along the last axis.

    Each value output depends on all input elements in its batch slice,
    since changing any input could change which elements are selected.
    The Jacobian of values is a selection matrix (one nonzero per row),
    but the sparsity pattern must cover all possible selections.
    The indices output is integer-valued and has zero derivative.

    For input shape (*batch, n) and parameter k:
        values[*b, j]  depends on all input[*b, :]
        indices[*b, j] has empty dependency sets

    Example: y_vals, y_idx = top_k(x, k=2) where x.shape = (2, 3)
        Input state_indices:  [{0}, {1}, {2}, {3}, {4}, {5}]
        Values state_indices: [{0, 1, 2}, {0, 1, 2}, {3, 4, 5}, {3, 4, 5}]
        Index state_indices:  [{}, {}, {}, {}]

    Jaxpr:
        invars[0]: input array
        k: number of top elements to select

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.top_k.html
    """
    in_indices = index_sets(state_indices, eqn.invars[0])
    in_shape = atom_shape(eqn.invars[0])
    k = eqn.params["k"]
    last_dim = in_shape[-1]
    n_batches = math.prod(in_shape[:-1]) if len(in_shape) > 1 else 1

    # Union input state_indices within each batch slice (contiguous along last axis)
    group_indices = [
        union_all(in_indices[b * last_dim : (b + 1) * last_dim])
        for b in range(n_batches)
    ]

    # Each of the k value outputs per batch copies its group's state_indices
    state_indices[eqn.outvars[0]] = [
        group_indices[b] for b in range(n_batches) for _ in range(k)
    ]
    state_indices[eqn.outvars[1]] = [set() for _ in range(atom_numel(eqn.outvars[1]))]
