"""Propagation rule for split operations."""

from jax._src.core import JaxprEqn

from ._commons import StateIndices, atom_shape, index_sets, transform_indices


def prop_split(eqn: JaxprEqn, state_indices: StateIndices) -> None:
    """Split partitions an array into multiple sub-arrays along an axis.

    Each output element maps to exactly one input element,
    so dependencies pass through unchanged.
    Output `k` contains elements from the slice
    ``[sum(sizes[:k]) : sum(sizes[:k]) + sizes[k]]`` along the split axis.

    The Jacobian is a permutation of rows of the identity matrix.

    Example: x = [a, b, c, d], split(x, sizes=(2, 2))
        Input state_indices:  [{0}, {1}, {2}, {3}]
        Output 0 state_indices: [{0}, {1}]
        Output 1 state_indices: [{2}, {3}]

    Jaxpr:
        invars[0]: input array
        axis: dimension along which to split
        sizes: tuple of sizes for each output chunk

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.split.html
    """
    in_indices = index_sets(state_indices, eqn.invars[0])
    in_shape = atom_shape(eqn.invars[0])
    axis = eqn.params["axis"]
    sizes = eqn.params["sizes"]
    ndim = len(in_shape)

    offset = 0
    for k, out_var in enumerate(eqn.outvars):
        size_k = sizes[k]
        slices = [slice(None)] * ndim
        slices[axis] = slice(offset, offset + size_k)
        state_indices[out_var] = transform_indices(
            in_indices, in_shape, lambda p, s=tuple(slices): p[s]
        )
        offset += size_k
