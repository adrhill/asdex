"""Propagation rule for dot_general (generalized matrix multiply)."""

import numpy as np
from jax._src.core import JaxprEqn

from ._common import (
    IndexSet,
    _atom_const_val,
    _atom_shape,
    _empty_index_sets,
    _index_sets,
    _numel,
    _PropState,
    _row_strides,
    _union_all,
)


def _fixed_base_positions(
    shape: tuple[int, ...], dims: tuple[int, ...], strides: tuple[int, ...]
) -> np.ndarray:
    """Flat operand positions of the zero contracting coordinate per fixed coordinate.

    ``dims`` lists the operand's batch and free dimensions
    in the order of the corresponding output axes,
    so the result enumerates the fixed coordinates in the same C order
    as the output axes they map to.
    Adding a contracting offset to a base yields a full flat operand position.
    """
    sizes = tuple(shape[d] for d in dims)
    coords = (
        np.indices(sizes, dtype=np.int64).reshape(len(dims), -1)
        if sizes
        else np.zeros((0, 1), dtype=np.int64)
    )
    bases = np.zeros(_numel(sizes), dtype=np.int64)
    for i, d in enumerate(dims):
        bases += coords[i] * strides[d]
    return bases


def _contract_union_sets(
    indices: list[IndexSet], bases: np.ndarray, offsets: list[int]
) -> list[IndexSet]:
    """Union the index sets over the contracting offsets for each base position.

    For lhs these are the row sets ``deps(lhs[b, i, :])``,
    for rhs the column sets ``deps(rhs[b, :, j])``.
    """
    return [_union_all([indices[base + o] for o in offsets]) for base in bases.tolist()]


def _one_const_indices(
    const_vals: np.ndarray,
    const_bases: np.ndarray,
    const_offsets: np.ndarray,
    traced_indices: list[IndexSet],
    traced_bases: np.ndarray,
    traced_offsets: np.ndarray,
    batch_size: int,
    n_const: int,
    n_traced: int,
    const_is_lhs: bool,
) -> list[IndexSet]:
    """Output index sets when exactly one operand is a statically known constant.

    The constant operand carries no input dependencies,
    so each output element unions the traced operand's index sets
    over the contracting positions where the constant is nonzero (zero-skipping).
    Fixed positions where the constant has no zeros share
    one unmasked union per traced fixed position.
    """
    n_contract = len(const_offsets)
    all_offsets = traced_offsets.tolist()
    out_indices: list[IndexSet] = []
    for b in range(batch_size):
        const_bs = const_bases[b * n_const : (b + 1) * n_const].tolist()
        traced_bs = traced_bases[b * n_traced : (b + 1) * n_traced].tolist()
        # Unmasked unions for this batch index, built once on first use
        # and shared across all constant positions without zeros.
        full: list[IndexSet] | None = None
        # block[c][t] is the output set for const position c and traced position t.
        block: list[list[IndexSet]] = []
        for cbase in const_bs:
            kept = np.flatnonzero(const_vals[cbase + const_offsets])
            if kept.size == n_contract:
                if full is None:
                    full = [
                        _union_all([traced_indices[tb + o] for o in all_offsets])
                        for tb in traced_bs
                    ]
                block.append(full)
            else:
                offsets = traced_offsets[kept].tolist()
                block.append(
                    [
                        _union_all([traced_indices[tb + o] for o in offsets])
                        for tb in traced_bs
                    ]
                )
        if const_is_lhs:
            # Output axes per batch are (const fixed, traced fixed).
            for row in block:
                out_indices.extend(row)
        else:
            # Output axes per batch are (traced fixed, const fixed): transpose.
            for t in range(n_traced):
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

    batch_size = _numel(tuple(lhs_shape[d] for d in lhs_batch))
    lhs_free_size = _numel(tuple(lhs_shape[d] for d in lhs_free))
    rhs_free_size = _numel(tuple(rhs_shape[d] for d in rhs_free))

    lhs_strides = _row_strides(lhs_shape)
    rhs_strides = _row_strides(rhs_shape)

    # Flat offsets of the contracting positions, shared by every fixed position.
    # contract_coords[i] runs over the i-th contracting axis, shared by lhs and
    # rhs since lhs_contract[i] pairs with rhs_contract[i] and has equal size.
    contract_sizes = tuple(lhs_shape[d] for d in lhs_contract)
    n_contract = _numel(contract_sizes)
    contract_coords = (
        np.indices(contract_sizes, dtype=np.int64).reshape(len(contract_sizes), -1)
        if contract_sizes
        else np.zeros((0, 1), dtype=np.int64)
    )
    lhs_offsets = np.zeros(n_contract, dtype=np.int64)
    for i, d in enumerate(lhs_contract):
        lhs_offsets += contract_coords[i] * lhs_strides[d]
    rhs_offsets = np.zeros(n_contract, dtype=np.int64)
    for i, d in enumerate(rhs_contract):
        rhs_offsets += contract_coords[i] * rhs_strides[d]

    lhs_bases = _fixed_base_positions(lhs_shape, lhs_batch + lhs_free, lhs_strides)
    rhs_bases = _fixed_base_positions(rhs_shape, rhs_batch + rhs_free, rhs_strides)

    out_indices: list[IndexSet]
    match (lhs_known, rhs_known):
        case (False, False):
            # Both operands are traced: no zero-skipping possible,
            # every output is one row set unioned with one column set.
            row_sets = _contract_union_sets(
                lhs_indices, lhs_bases, lhs_offsets.tolist()
            )
            col_sets = _contract_union_sets(
                rhs_indices, rhs_bases, rhs_offsets.tolist()
            )
            out_indices = []
            for b in range(batch_size):
                rows = row_sets[b * lhs_free_size : (b + 1) * lhs_free_size]
                cols = col_sets[b * rhs_free_size : (b + 1) * rhs_free_size]
                for row in rows:
                    out_indices.extend(row | col for col in cols)
        case (True, False):
            assert lhs_val_flat is not None
            out_indices = _one_const_indices(
                lhs_val_flat,
                lhs_bases,
                lhs_offsets,
                rhs_indices,
                rhs_bases,
                rhs_offsets,
                batch_size,
                lhs_free_size,
                rhs_free_size,
                const_is_lhs=True,
            )
        case (False, True):
            assert rhs_val_flat is not None
            out_indices = _one_const_indices(
                rhs_val_flat,
                rhs_bases,
                rhs_offsets,
                lhs_indices,
                lhs_bases,
                lhs_offsets,
                batch_size,
                rhs_free_size,
                lhs_free_size,
                const_is_lhs=False,
            )
        case (True, True):
            # Both operands are statically known,
            # so no output element depends on the traced inputs.
            out_indices = _empty_index_sets(out_size)

    state.indices[eqn.outvars[0]] = out_indices
