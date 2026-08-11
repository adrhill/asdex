"""Handlers for linear algebra primitives."""

from jax._src.core import JaxprEqn

from ._common import (
    IndexSet,
    _atom_shape,
    _empty_index_set,
    _index_sets,
    _numel,
    _PropState,
    _report_issue,
    _union_all,
)


def _prop_qr(eqn: JaxprEqn, state: _PropState) -> None:
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
    # pivoting=True adds a permutation output this handler does not model.
    if len(eqn.outvars) != 2:
        msg = _report_issue(
            f"'qr' handler expects two outputs (Q, R), got {len(eqn.outvars)}."
        )
        raise NotImplementedError(msg)
    q_var, r_var = eqn.outvars

    # Collect all input index sets
    in_indices = _index_sets(state, invar)
    combined = _union_all(in_indices)

    # Q: all elements depend on all inputs
    state.indices[q_var] = [combined] * _numel(_atom_shape(q_var))

    # R: upper triangular
    # Upper triangle (including diagonal) depends on all inputs
    # Lower triangle is always 0 (no dependencies)
    r_shape = _atom_shape(r_var)

    if _numel(r_shape) == 0:
        state.indices[r_var] = []
        return

    nrows, ncols = r_shape[-2], r_shape[-1]
    batch_size = _numel(r_shape[:-2])

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

    state.indices[r_var] = r_indices
