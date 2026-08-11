"""Propagation rule for iota."""

import numpy as np
from jax._src.core import JaxprEqn

from ._common import _empty_index_sets, _numel, _PropState


def _prop_iota(eqn: JaxprEqn, state: _PropState) -> None:
    """Iota generates a constant index array with no input dependencies.

    The output is fully determined by the parameters (shape, dtype, dimension),
    so all dependency sets are empty.
    We also track the concrete values for downstream gather/scatter precision.

    Jaxpr:
        invars: [] (no inputs)
        shape: output shape
        dtype: output dtype
        dimension: axis along which indices increase

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.iota.html
    """
    shape = eqn.params["shape"]
    state.indices[eqn.outvars[0]] = _empty_index_sets(_numel(shape))

    dtype = eqn.params["dtype"]
    dim = eqn.params["dimension"]
    state.consts[eqn.outvars[0]] = np.broadcast_to(
        np.arange(shape[dim], dtype=dtype).reshape(
            [shape[dim] if i == dim else 1 for i in range(len(shape))]
        ),
        shape,
    )
