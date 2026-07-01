"""Verification utilities for checking asdex results against JAX references."""

from collections.abc import Callable
from typing import Any, Literal, assert_never

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO
from numpy.typing import ArrayLike, NDArray

from asdex._defaults import (
    _DEFAULT_NUM_PROBES,
    _DEFAULT_SEED,
    _DEFAULT_TOL,
    _DEFAULT_VERIFY_METHOD,
)
from asdex._docstrings import _fill_doc
from asdex._errors import InvalidColoringError, VerificationError
from asdex._pattern import ColoredPattern, SparsityPattern
from asdex._pytree import (
    _flatten_pytree,
    _output_size,
    _unflatten_to_pytree,
)
from asdex._types import _assert_jacobian_mode
from asdex.decompression import hessian_from_coloring, jacobian_from_coloring

# Coloring validators


def check_coloring_rows(sparsity: SparsityPattern, colors: NDArray[np.int32]) -> None:
    """Check a row coloring: no column contains two rows with the same color.

    Args:
        sparsity: Jacobian sparsity pattern of shape ``(m, n)``.
        colors: Row color assignment, shape ``(m,)``.

    Raises:
        InvalidColoringError: If the coloring is invalid.
    """
    for j, rows_in_col in sparsity.col_to_rows.items():
        colors_in_col = [int(colors[r]) for r in rows_in_col if colors[r] >= 0]
        if len(colors_in_col) != len(set(colors_in_col)):
            msg = (
                f"Invalid row coloring: column {j} contains two rows with the "
                f"same color (rows {rows_in_col}, colors {colors_in_col})"
            )
            raise InvalidColoringError(msg)


def check_coloring_cols(sparsity: SparsityPattern, colors: NDArray[np.int32]) -> None:
    """Check a column coloring: no row contains two columns with the same color.

    Args:
        sparsity: Jacobian sparsity pattern of shape ``(m, n)``.
        colors: Column color assignment, shape ``(n,)``.

    Raises:
        InvalidColoringError: If the coloring is invalid.
    """
    for i, cols_in_row in sparsity.row_to_cols.items():
        colors_in_row = [int(colors[c]) for c in cols_in_row if colors[c] >= 0]
        if len(colors_in_row) != len(set(colors_in_row)):
            msg = (
                f"Invalid column coloring: row {i} contains two columns with the "
                f"same color (cols {cols_in_row}, colors {colors_in_row})"
            )
            raise InvalidColoringError(msg)


def check_coloring_symmetric(
    sparsity: SparsityPattern, colors: NDArray[np.int32]
) -> None:
    """Check a star coloring: distance-1 coloring + no 2-colored path of 4 vertices.

    A star coloring satisfies:

    1. Adjacent vertices have different colors (distance-1).
    2. Every path on 4 vertices uses at least 3 distinct colors
       (no 2-colored P4).

    Vertices with color ``-1`` (neutral, from postprocessing) are excluded
    from the P4 check as they represent the absence of a color.

    Args:
        sparsity: Hessian sparsity pattern of shape ``(n, n)``.
        colors: Vertex color assignment, shape ``(n,)``.

    Raises:
        ValueError: If pattern is not square.
        InvalidColoringError: If the coloring is invalid.
    """
    if sparsity.m != sparsity.n:
        msg = f"Star coloring requires a square pattern, got shape {sparsity.shape}"
        raise ValueError(msg)

    n = sparsity.n

    adj: list[set[int]] = [set() for _ in range(n)]
    for i, j in zip(sparsity.rows, sparsity.cols, strict=True):
        i_int, j_int = int(i), int(j)
        if i_int != j_int:
            adj[i_int].add(j_int)
            adj[j_int].add(i_int)

    # Distance-1: adjacent vertices must have different colors (both active).
    for v in range(n):
        cv = int(colors[v])
        if cv < 0:
            continue
        for w in adj[v]:
            cw = int(colors[w])
            if cw >= 0 and cw == cv:
                msg = (
                    f"Invalid star coloring: adjacent vertices {v} and {w} "
                    f"share color {cv}"
                )
                raise InvalidColoringError(msg)

    # No 2-colored P4: for every path v0-v1-v2-v3, |{colors}| >= 3.
    for v1 in range(n):
        for v2 in adj[v1]:
            if v2 <= v1:
                continue
            for v0 in adj[v1]:
                if v0 == v2:
                    continue
                for v3 in adj[v2]:
                    if v3 == v1:
                        continue
                    path = (
                        int(colors[v0]),
                        int(colors[v1]),
                        int(colors[v2]),
                        int(colors[v3]),
                    )
                    if any(c < 0 for c in path):
                        continue
                    if len(set(path)) < 3:
                        msg = (
                            f"Invalid star coloring: 2-colored P4 at vertices "
                            f"{v0}-{v1}-{v2}-{v3} with colors {path}"
                        )
                        raise InvalidColoringError(msg)


@_fill_doc
def check_jacobian_correctness(
    f: Callable[..., Any],
    x: Any,
    coloring: ColoredPattern,
    *,
    method: Literal["matvec", "dense"] = _DEFAULT_VERIFY_METHOD,
    num_probes: int = _DEFAULT_NUM_PROBES,
    seed: int = _DEFAULT_SEED,
    rtol: float | None = _DEFAULT_TOL,
    atol: float | None = _DEFAULT_TOL,
) -> None:
    """Verify asdex's sparse Jacobian against a JAX reference at a given input.

    Args:
        f: Function whose Jacobian is to be verified.
        x: Input at which to evaluate the Jacobian.
            For multi-input functions (where ``argnums`` is a tuple),
            pass a tuple of all positional arguments.
        coloring: Pre-computed colored pattern from
            :func:`~asdex.jacobian_coloring`.
        method: {verify_method}
        num_probes: {num_probes}
        seed: {seed}
        rtol: {rtol}
        atol: {atol}

    Raises:
        VerificationError: If the sparse and reference Jacobians disagree.
    """
    if method not in ("matvec", "dense"):
        raise ValueError(f"Unknown method {method!r}. Expected 'matvec' or 'dense'.")

    # Derive reference AD mode from the colored pattern
    _assert_jacobian_mode(coloring.mode)
    match coloring.mode:
        case "fwd" | "rev":
            ref_mode = coloring.mode
        case _ as unreachable:
            assert_never(unreachable)  # ty: ignore[type-assertion-failure]

    sparsity = coloring.sparsity
    argnums = sparsity.argnums
    multi_input = isinstance(argnums, tuple)

    if multi_input:
        args = x
        out_size = _output_size(jax.eval_shape(f, *args))
    else:
        args = (x,)
        out_size = _output_size(jax.eval_shape(f, x))

    if out_size != sparsity.m:
        raise VerificationError(
            f"asdex's sparse Jacobian output size {sparsity.m} does not "
            f"match the shape of f(x), which has {out_size} elements. "
            "This likely means the detected sparsity pattern is missing nonzeros. "
            "Please help out asdex's development by reporting this at "
            "https://github.com/adrhill/asdex/issues"
        )

    match method:
        case "dense":
            jac_fn = jax.jacfwd if ref_mode == "fwd" else jax.jacrev
            J_asdex = jacobian_from_coloring(f, coloring, output_format="dense")(*args)
            J_ref = jac_fn(f, argnums=argnums)(*args)
            rtol_ = rtol if rtol is not None else 1e-7
            atol_ = atol if atol is not None else 1e-7
            if not _allclose_pytree(J_asdex, J_ref, rtol=rtol_, atol=atol_):
                raise VerificationError(
                    "asdex's sparse Jacobian does not match JAX's dense reference. "
                    "This likely means the detected sparsity pattern is missing nonzeros. "
                    "Please help out asdex's development by reporting this at "
                    "https://github.com/adrhill/asdex/issues"
                )
        case "matvec":
            J_sparse = jacobian_from_coloring(f, coloring)(*args)
            _check_jacobian_matvec(
                f,
                x,
                J_sparse,
                sparsity=sparsity,
                ref_mode=ref_mode,
                num_probes=num_probes,
                seed=seed,
                rtol=rtol,
                atol=atol,
            )
        case _ as unreachable:
            assert_never(unreachable)


@_fill_doc
def check_hessian_correctness(
    f: Callable[..., Any],
    x: Any,
    coloring: ColoredPattern,
    *,
    method: Literal["matvec", "dense"] = _DEFAULT_VERIFY_METHOD,
    num_probes: int = _DEFAULT_NUM_PROBES,
    seed: int = _DEFAULT_SEED,
    rtol: float | None = _DEFAULT_TOL,
    atol: float | None = _DEFAULT_TOL,
) -> None:
    """Verify asdex's sparse Hessian against a JAX reference at a given input.

    Args:
        f: Scalar-valued function taking an array.
        x: Input at which to evaluate the Hessian.
            For multi-input functions (where ``argnums`` is a tuple),
            pass a tuple of all positional arguments.
        coloring: Pre-computed colored pattern from
            :func:`~asdex.hessian_coloring`.
        method: {verify_method}
        num_probes: {num_probes}
        seed: {seed}
        rtol: {rtol}
        atol: {atol}

    Raises:
        VerificationError: If the sparse and reference Hessians disagree.
    """
    if method not in ("matvec", "dense"):
        raise ValueError(f"Unknown method {method!r}. Expected 'matvec' or 'dense'.")

    # Derive reference AD mode from the colored pattern
    match coloring.mode:
        case "fwd_over_rev" | "rev_over_fwd" | "rev_over_rev":
            hessian_mode = coloring.mode
        case "fwd" | "rev":
            raise ValueError(f"Expected a Hessian mode, got {coloring.mode!r}.")
        case _ as unreachable:
            assert_never(unreachable)

    sparsity = coloring.sparsity
    argnums = sparsity.argnums
    multi_input = isinstance(argnums, tuple)
    args = x if multi_input else (x,)

    match method:
        case "dense":
            H_asdex = hessian_from_coloring(f, coloring, output_format="dense")(*args)
            H_ref = jax.hessian(f, argnums=argnums)(*args)
            rtol_ = rtol if rtol is not None else 1e-7
            atol_ = atol if atol is not None else 1e-7
            if not _allclose_pytree(H_asdex, H_ref, rtol=rtol_, atol=atol_):
                raise VerificationError(
                    "asdex's sparse Hessian does not match JAX's dense reference. "
                    "This likely means the detected sparsity pattern is missing nonzeros. "
                    "Please help out asdex's development by reporting this at "
                    "https://github.com/adrhill/asdex/issues"
                )
        case "matvec":
            H_sparse = hessian_from_coloring(f, coloring)(*args)
            _check_hessian_matvec(
                f,
                x,
                H_sparse,
                sparsity=sparsity,
                hessian_mode=hessian_mode,
                num_probes=num_probes,
                seed=seed,
                rtol=rtol,
                atol=atol,
            )
        case _ as unreachable:
            assert_never(unreachable)


# Private helpers


def _is_bcoo(x: Any) -> bool:
    return isinstance(x, BCOO)


def _flatten_jacobian_to_dense(
    J_pytree: Any,
    sparsity: SparsityPattern,
    out_struct: Any,
) -> jax.Array:
    """Flatten a PyTree of Jacobian blocks into a (m, n) dense matrix.

    This mirrors decompression._assemble_jacobian_blocks in reverse:
    that function splits a (m, n) matrix into blocks, this reassembles them.

    Args:
        J_pytree: PyTree of BCOO/array blocks with structure (out_tree, in_tree).
        sparsity: SparsityPattern providing input leaf shapes/sizes.
        out_struct: Output structure from jax.eval_shape(f, *args).

    Returns:
        Dense (m, n) array where m = output size, n = selected input size.
    """
    in_leaf_sizes = sparsity.leaf_sizes

    out_leaves = jax.tree_util.tree_leaves(out_struct)
    out_leaf_sizes = [int(np.prod(leaf.shape)) for leaf in out_leaves]

    m = sum(out_leaf_sizes)
    n = sum(in_leaf_sizes)
    dense = jnp.zeros((m, n))

    # J_pytree has structure (out_tree, in_tree) - flatten to get blocks
    # in the same order as _assemble_jacobian_blocks produces them
    out_row_offset = 0
    for out_idx, out_size in enumerate(out_leaf_sizes):
        in_col_offset = 0
        for in_idx, in_size in enumerate(in_leaf_sizes):
            # Navigate to the block: J_pytree[out_key][in_key] for dicts
            block = _get_jacobian_block(J_pytree, out_idx, in_idx, out_struct, sparsity)
            block_dense = block.todense() if _is_bcoo(block) else jnp.asarray(block)
            block_2d = block_dense.reshape(out_size, in_size)
            dense = dense.at[
                out_row_offset : out_row_offset + out_size,
                in_col_offset : in_col_offset + in_size,
            ].set(block_2d)
            in_col_offset += in_size
        out_row_offset += out_size

    return dense


def _get_jacobian_block(
    J_pytree: Any,
    out_idx: int,
    in_idx: int,
    out_struct: Any,
    sparsity: SparsityPattern,
) -> Any:
    """Extract a single Jacobian block from the nested PyTree structure."""
    # Get the output-indexed subtree
    out_leaves, _ = jax.tree_util.tree_flatten(out_struct)
    if len(out_leaves) == 1:
        # Single output leaf - J_pytree is just the input tree
        out_subtree = J_pytree
    else:
        # Multiple output leaves - J_pytree[out_key] gives input tree
        out_keys = _get_sorted_keys(J_pytree)
        out_subtree = J_pytree[out_keys[out_idx]]

    # Get the input-indexed block from the output subtree
    in_leaves = jax.tree_util.tree_leaves(out_subtree, is_leaf=_is_bcoo)
    return in_leaves[in_idx]


def _get_sorted_keys(pytree: Any) -> list[Any]:
    """Get keys from a PyTree container in JAX's canonical order."""
    if isinstance(pytree, dict):
        return sorted(pytree.keys())
    if isinstance(pytree, (list, tuple)):
        return list(range(len(pytree)))
    msg = f"Unsupported PyTree type: {type(pytree)}"
    raise TypeError(msg)


def _stack_hessian_pytree(pytree: Any, n: int) -> BCOO:
    """Stack a PyTree-of-PyTrees of BCOO matrices into a single (n, n) BCOO.

    The outer structure represents rows, inner structure represents columns.
    We concatenate column blocks horizontally, then stack rows vertically.
    """
    leaves = jax.tree_util.tree_leaves(pytree, is_leaf=_is_bcoo)
    if len(leaves) == 1:
        return leaves[0]

    # Get outer children (row groups)
    outer_children, _ = jax.tree_util.tree_flatten(pytree, is_leaf=_is_bcoo)
    if all(isinstance(c, BCOO) for c in outer_children):
        # Flat list of BCOOs - determine grid dimensions
        num_leaves = len(outer_children)
        side = int(num_leaves**0.5)
        rows = []
        for i in range(side):
            row_blocks = [outer_children[i * side + j].todense() for j in range(side)]
            rows.append(jnp.concatenate(row_blocks, axis=1))
        stacked = jnp.concatenate(rows, axis=0)
        return BCOO.fromdense(stacked)

    # Nested structure - each outer child is a row of blocks
    rows = []
    for row_child in outer_children:
        col_blocks = jax.tree_util.tree_leaves(row_child, is_leaf=_is_bcoo)
        col_dense = [b.todense() for b in col_blocks]
        row = jnp.concatenate(col_dense, axis=1)
        rows.append(row)

    stacked = jnp.concatenate(rows, axis=0)
    return BCOO.fromdense(stacked)


def _check_jacobian_matvec(
    f: Callable[..., Any],
    x: Any,
    J_sparse: Any,
    *,
    sparsity: SparsityPattern,
    ref_mode: Literal["fwd", "rev"],
    num_probes: int,
    seed: int,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    """Verify a sparse Jacobian via randomized matvec products."""
    rtol = rtol if rtol is not None else 1e-5
    atol = atol if atol is not None else 1e-5
    key = jax.random.key(seed)
    keys = jax.random.split(key, num_probes)

    multi_input = isinstance(sparsity.argnums, tuple)
    args = x if multi_input else (x,)

    out_struct = jax.eval_shape(f, *args)
    m = _output_size(out_struct)

    dyn_args = tuple(args[i] for i in sparsity._argnums_tuple)
    n = sum(_output_size(a) for a in dyn_args)

    # Flatten PyTree of BCOOs into a single (m, n) matrix if needed
    if not isinstance(J_sparse, BCOO):
        J_dense = _flatten_jacobian_to_dense(J_sparse, sparsity, out_struct)
        J_sparse = BCOO.fromdense(J_dense)

    for i in range(num_probes):
        match ref_mode:
            case "fwd":
                v = jax.random.normal(keys[i], shape=(n,))
                sparse_result = (J_sparse @ v).ravel()
                tangent = _unflatten_to_pytree(v, sparsity.example_input)
                if multi_input:
                    tangents = tuple(
                        tangent[j]
                        if j in sparsity._argnums_tuple
                        else jax.tree.map(jnp.zeros_like, args[j])
                        for j in range(len(args))
                    )
                    _, ref_result = jax.jvp(f, args, tangents)
                else:
                    _, ref_result = jax.jvp(f, args, (tangent,))
                ref_result = _flatten_pytree(ref_result)
            case "rev":
                v = jax.random.normal(keys[i], shape=(m,))
                sparse_result = (v @ J_sparse).ravel()
                _, vjp_fn = jax.vjp(f, *args)
                cotangent = _unflatten_to_pytree(v, out_struct)
                ref_results = vjp_fn(cotangent)
                selected = tuple(ref_results[j] for j in sparsity._argnums_tuple)
                ref_result = _flatten_pytree(selected if multi_input else selected[0])
            case _ as unreachable:
                assert_never(unreachable)

        _check_matvec_allclose(
            sparse_result,
            ref_result,
            "Jacobian",
            probe=i,
            num_probes=num_probes,
            rtol=rtol,
            atol=atol,
        )


def _check_hessian_matvec(
    f: Callable[[ArrayLike], ArrayLike],
    x: Any,
    H_sparse: Any,
    *,
    sparsity: SparsityPattern,
    hessian_mode: Literal["fwd_over_rev", "rev_over_fwd", "rev_over_rev"],
    num_probes: int,
    seed: int,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    """Verify a sparse Hessian via randomized H @ v products."""
    rtol = rtol if rtol is not None else 1e-5
    atol = atol if atol is not None else 1e-5

    multi_input = isinstance(sparsity.argnums, tuple)
    args = x if multi_input else (x,)
    dyn_args = tuple(args[i] for i in sparsity._argnums_tuple)
    n = sum(_output_size(a) for a in dyn_args)

    key = jax.random.key(seed)
    keys = jax.random.split(key, num_probes)

    # Stack PyTree-of-PyTrees of BCOOs into a single (n, n) matrix if needed
    if not isinstance(H_sparse, BCOO):
        H_sparse = _stack_hessian_pytree(H_sparse, n)

    def unflatten_tangent(v: jax.Array) -> Any:
        return _unflatten_to_pytree(v, sparsity.example_input)

    argnums = sparsity.argnums

    match hessian_mode:
        case "fwd_over_rev":

            def hvp(v: jax.Array) -> jax.Array:
                tangent = unflatten_tangent(v)
                if multi_input:
                    tangents = tuple(
                        tangent[j]
                        if j in sparsity._argnums_tuple
                        else jax.tree.map(jnp.zeros_like, args[j])
                        for j in range(len(args))
                    )
                    _, result = jax.jvp(jax.grad(f, argnums=argnums), args, tangents)
                else:
                    _, result = jax.jvp(jax.grad(f), args, (tangent,))
                return _flatten_pytree(result)

        case "rev_over_fwd":

            def hvp(v: jax.Array) -> jax.Array:
                tangent = unflatten_tangent(v)
                if multi_input:
                    tangents = tuple(
                        tangent[j]
                        if j in sparsity._argnums_tuple
                        else jax.tree.map(jnp.zeros_like, args[j])
                        for j in range(len(args))
                    )

                    def fwd_fn(*ps: Any) -> jax.Array:
                        return jax.jvp(f, ps, tangents)[1]

                    result = jax.grad(fwd_fn, argnums=argnums)(*args)
                else:
                    result = jax.grad(lambda p: jax.jvp(f, (p,), (tangent,))[1])(*args)
                return _flatten_pytree(result)

        case "rev_over_rev":

            def hvp(v: jax.Array) -> jax.Array:
                tangent = unflatten_tangent(v)

                def inner(*ys: Any) -> jax.Array:
                    grads = jax.grad(f, argnums=argnums)(*ys)
                    if multi_input:
                        grad_leaves = jax.tree_util.tree_leaves(grads)
                        tangent_leaves = jax.tree_util.tree_leaves(tangent)
                    else:
                        grad_leaves = jax.tree_util.tree_leaves(grads)
                        tangent_leaves = jax.tree_util.tree_leaves(tangent)
                    dots = [
                        jnp.vdot(g, t)
                        for g, t in zip(grad_leaves, tangent_leaves, strict=True)
                    ]
                    return jnp.sum(jnp.stack(dots))

                result = jax.grad(inner, argnums=argnums)(*args)
                return _flatten_pytree(result)

        case _ as unreachable:
            assert_never(unreachable)

    for i in range(num_probes):
        v = jax.random.normal(keys[i], shape=(n,))
        sparse_result = (H_sparse @ v).ravel()
        ref_result = hvp(v)
        _check_matvec_allclose(
            sparse_result,
            ref_result,
            "Hessian",
            probe=i,
            num_probes=num_probes,
            rtol=rtol,
            atol=atol,
        )


def _check_matvec_allclose(
    sparse_result: jax.Array,
    ref_result: jax.Array,
    name: str,
    *,
    probe: int,
    num_probes: int,
    rtol: float,
    atol: float,
) -> None:
    """Compare a sparse matvec against a reference, raising on mismatch."""
    sparse_np = np.asarray(sparse_result)
    ref_np = np.asarray(ref_result)

    try:
        np.testing.assert_allclose(sparse_np, ref_np, rtol=rtol, atol=atol)
    except AssertionError:
        raise VerificationError(
            f"asdex's sparse {name} failed randomized matvec verification "
            f"(probe {probe + 1}/{num_probes}). "
            "This likely means the detected sparsity pattern is missing nonzeros. "
            "Please help out asdex's development by reporting this at "
            "https://github.com/adrhill/asdex/issues"
        ) from None


def _check_allclose(
    sparse: jax.Array,
    dense: jax.Array,
    name: str,
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    """Compare sparse and dense results, raising VerificationError on mismatch."""
    rtol = rtol if rtol is not None else 1e-7
    atol = atol if atol is not None else 1e-7
    sparse_np = np.asarray(sparse)
    dense_np = np.asarray(dense)

    if sparse_np.shape != dense_np.shape:
        raise VerificationError(
            f"asdex's sparse {name} has shape {sparse_np.shape} "
            f"but JAX's dense reference has shape {dense_np.shape}. "
            "This likely means the detected sparsity pattern is missing nonzeros. "
            "Please help out asdex's development by reporting this at "
            "https://github.com/adrhill/asdex/issues"
        )

    try:
        np.testing.assert_allclose(sparse_np, dense_np, rtol=rtol, atol=atol)
    except AssertionError:
        raise VerificationError(
            f"asdex's sparse {name} does not match JAX's dense reference. "
            "This likely means the detected sparsity pattern is missing nonzeros. "
            "Please help out asdex's development by reporting this at "
            "https://github.com/adrhill/asdex/issues"
        ) from None


def _allclose_pytree(
    a: Any,
    b: Any,
    rtol: float = 1e-7,
    atol: float = 1e-7,
) -> bool:
    """Check if two PyTrees are element-wise equal within tolerance."""

    def allclose_leaf(a_leaf: jax.Array, b_leaf: jax.Array) -> bool:
        return bool(jnp.allclose(a_leaf, b_leaf, rtol=rtol, atol=atol))

    results = jax.tree.map(allclose_leaf, a, b)
    return all(jax.tree_util.tree_leaves(results))
