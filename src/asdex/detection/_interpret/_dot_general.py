"""Propagation rule for dot_general (generalized matrix multiply)."""

import numpy as np
from jax._src.core import JaxprEqn

from ._common import (
    IndexSet,
    _atom_const_val,
    _atom_shape,
    _dim_offsets,
    _empty_index_sets,
    _index_sets,
    _numel,
    _PropState,
    _union_all,
)


def _contract_union_sets(
    indices: list[IndexSet], bases: np.ndarray, offsets: np.ndarray
) -> list[IndexSet]:
    """Union the index sets over the contracting offsets for each base position.

    For lhs these are the row sets, the unioned index sets of ``lhs[b, i, :]``.
    For rhs the column sets, the unioned index sets of ``rhs[b, :, j]``.
    """
    offset_list = offsets.tolist()
    return [
        _union_all([indices[base + o] for o in offset_list]) for base in bases.tolist()
    ]


def _zero_skipping_index_sets(
    *,
    const_vals: np.ndarray,
    const_bases: np.ndarray,
    const_offsets: np.ndarray,
    traced_indices: list[IndexSet],
    traced_bases: np.ndarray,
    traced_offsets: np.ndarray,
    const_is_lhs: bool,
) -> list[IndexSet]:
    """Output index sets of dot_general when exactly one operand is a known constant.

    The constant operand carries no input dependencies,
    so each output element unions the traced operand's index sets
    over the contracting positions where the constant is nonzero (zero-skipping).
    Fixed positions where the constant has no zeros share
    one unmasked union per traced fixed position.

    Args:
        const_vals: Flat values of the constant operand.
        const_bases: Constant operand bases from `_dim_offsets` over its batch and free dims,
            of shape ``(batch_size, const_free_size)``.
        const_offsets: Flat contracting offsets into the constant operand.
        traced_indices: Flat index sets of the traced operand.
        traced_bases: Traced operand bases from `_dim_offsets` over its batch and free dims,
            of shape ``(batch_size, traced_free_size)``.
        traced_offsets: Flat contracting offsets into the traced operand,
            paired elementwise with ``const_offsets``.
        const_is_lhs: Whether the constant is the lhs operand.
            The output axes are always (batch, lhs free, rhs free),
            so this decides whether the constant's free axes come first.

    Returns:
        Flat output index sets in C order of the dot_general output.
    """
    n_contract = len(const_offsets)
    out_indices: list[IndexSet] = []
    for const_bs, traced_bs in zip(const_bases, traced_bases, strict=True):
        # Unmasked unions for this batch index, built once on first use
        # and shared across all constant positions without zeros.
        full: list[IndexSet] | None = None
        # block[c][t] is the output set for const position c and traced position t.
        block: list[list[IndexSet]] = []
        for cbase in const_bs.tolist():
            kept = np.flatnonzero(const_vals[cbase + const_offsets])
            if kept.size == n_contract:
                if full is None:
                    full = _contract_union_sets(
                        traced_indices, traced_bs, traced_offsets
                    )
                block.append(full)
            else:
                block.append(
                    _contract_union_sets(
                        traced_indices, traced_bs, traced_offsets[kept]
                    )
                )
        if const_is_lhs:
            # Output axes per batch are (const fixed, traced fixed).
            for row in block:
                out_indices.extend(row)
        else:
            # Output axes per batch are (traced fixed, const fixed): transpose.
            for t in range(len(traced_bs)):
                out_indices.extend(row[t] for row in block)
    return out_indices


def _prop_dot_general(eqn: JaxprEqn, state: _PropState) -> None:
    """Dot_general contracts and batches two arrays.

    Each output element is a sum of products over the contracting dimensions,
    so it depends on a slice of lhs and a slice of rhs.
    Batch dimensions are preserved one-to-one.

    For out[b..., i..., j...] = sum_k lhs[b..., i..., k...] * rhs[b..., k..., j...]:
        indices(out[b,i,j]) = indices(lhs[b, i, :]) | indices(rhs[b, :, j])
    where b are batch dims, i are lhs-free dims, j are rhs-free dims,
    and k are contracting dims.

    Because union distributes over the sum of products,
    the union over contraction terms factors into
    a row union of lhs and a column union of rhs.
    Both are precomputed once per fixed (batch and free) position,
    so each output element costs a single union of two sets
    instead of one union per contraction term.

    Example: matrix multiply A(2,3) @ B(3,4) -> C(2,4)
        contracting: lhs_dim=1, rhs_dim=0
        out[i,j] depends on lhs[i,:] and rhs[:,j]
        Input lhs index sets:  [{0},{1},{2},{3},{4},{5}]  (shape 2x3)
        Input rhs index sets:  [{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17}]
        Output state.indices[0,0] = {0,1,2} | {6,10,14} = {0,1,2,6,10,14}

    Zero-skipping: when an operand is a statically known constant with zeros,
    the contracting positions where that factor is zero contribute nothing
    to the derivative and are dropped from the pattern.
    A statically known operand carries no input dependencies itself,
    so only the other operand contributes index sets in that case.

    Jaxpr:
        invars[0]: lhs array
        invars[1]: rhs array
        dimension_numbers: ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch))

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.dot_general.html
    """
    lhs_indices = _index_sets(state, eqn.invars[0])
    rhs_indices = _index_sets(state, eqn.invars[1])

    lhs_shape = _atom_shape(eqn.invars[0])
    rhs_shape = _atom_shape(eqn.invars[1])

    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = eqn.params[
        "dimension_numbers"
    ]
    lhs_contract = tuple(lhs_contract)
    rhs_contract = tuple(rhs_contract)
    lhs_batch = tuple(lhs_batch)
    rhs_batch = tuple(rhs_batch)

    lhs_free = tuple(
        d for d in range(len(lhs_shape)) if d not in lhs_contract and d not in lhs_batch
    )
    rhs_free = tuple(
        d for d in range(len(rhs_shape)) if d not in rhs_contract and d not in rhs_batch
    )

    # Output dim order: batch, lhs_free, rhs_free.
    out_shape = (
        tuple(lhs_shape[d] for d in lhs_batch)
        + tuple(lhs_shape[d] for d in lhs_free)
        + tuple(rhs_shape[d] for d in rhs_free)
    )
    out_size = _numel(out_shape)

    if out_size == 0:
        # Zero-sized output has no elements to depend on anything.
        state.indices[eqn.outvars[0]] = []
        return

    # Get constant values for zero-skipping.
    # When an operand is a known constant with zeros,
    # those contracting positions contribute nothing to the derivative.
    lhs_val = _atom_const_val(eqn.invars[0], state)
    rhs_val = _atom_const_val(eqn.invars[1], state)
    lhs_val_flat = np.atleast_1d(lhs_val).ravel() if lhs_val is not None else None
    rhs_val_flat = np.atleast_1d(rhs_val).ravel() if rhs_val is not None else None

    # When a constant was scalar-broadcast to a larger shape
    # (e.g. jnp.dot(jnp.array(2.0), x)), expand it to full size
    # so zero-skipping still works for scalar constants like 0.0.
    if lhs_val_flat is not None and len(lhs_val_flat) != _numel(lhs_shape):
        lhs_val_flat = np.broadcast_to(lhs_val_flat, _numel(lhs_shape))
    if rhs_val_flat is not None and len(rhs_val_flat) != _numel(rhs_shape):
        rhs_val_flat = np.broadcast_to(rhs_val_flat, _numel(rhs_shape))

    # A statically known operand carries no input dependencies.
    # Should an operand ever have both a known value and dependencies,
    # ignore the value and treat the operand as traced,
    # which keeps the pattern conservative instead of dropping dependencies.
    lhs_known = lhs_val_flat is not None and not any(lhs_indices)
    rhs_known = rhs_val_flat is not None and not any(rhs_indices)

    # Flat offsets of the contracting positions, shared by every fixed position.
    # Both sides enumerate the contracting coordinates in the same C order,
    # since lhs_contract[i] pairs with rhs_contract[i] and has equal size.
    lhs_offsets = _dim_offsets(lhs_shape, lhs_contract)
    rhs_offsets = _dim_offsets(rhs_shape, rhs_contract)

    # Flat positions of the fixed (batch and free) coordinates at contracting zero.
    # Listing batch dims before free dims enumerates them
    # in the same C order as the output axes they map to.
    # Shape (batch_size, free_size) each, with matching batch rows.
    batch_size = _numel(tuple(lhs_shape[d] for d in lhs_batch))
    lhs_bases = _dim_offsets(lhs_shape, lhs_batch + lhs_free).reshape(batch_size, -1)
    rhs_bases = _dim_offsets(rhs_shape, rhs_batch + rhs_free).reshape(batch_size, -1)

    out_indices: list[IndexSet]
    match (lhs_known, rhs_known):
        case (False, False):
            # Both operands are traced: no zero-skipping possible,
            # every output is one row set unioned with one column set.
            out_indices = []
            for lhs_bs, rhs_bs in zip(lhs_bases, rhs_bases, strict=True):
                rows = _contract_union_sets(lhs_indices, lhs_bs, lhs_offsets)
                cols = _contract_union_sets(rhs_indices, rhs_bs, rhs_offsets)
                for row in rows:
                    out_indices.extend(row | col for col in cols)
        case (True, False):
            assert lhs_val_flat is not None
            out_indices = _zero_skipping_index_sets(
                const_vals=lhs_val_flat,
                const_bases=lhs_bases,
                const_offsets=lhs_offsets,
                traced_indices=rhs_indices,
                traced_bases=rhs_bases,
                traced_offsets=rhs_offsets,
                const_is_lhs=True,
            )
        case (False, True):
            assert rhs_val_flat is not None
            out_indices = _zero_skipping_index_sets(
                const_vals=rhs_val_flat,
                const_bases=rhs_bases,
                const_offsets=rhs_offsets,
                traced_indices=lhs_indices,
                traced_bases=lhs_bases,
                traced_offsets=lhs_offsets,
                const_is_lhs=False,
            )
        case (True, True):
            # Both operands are statically known,
            # so no output element depends on the traced inputs.
            out_indices = _empty_index_sets(out_size)

    state.indices[eqn.outvars[0]] = out_indices
