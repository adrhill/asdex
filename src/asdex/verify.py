"""Verification utilities for checking asdex results against JAX references."""

from collections.abc import Callable
from typing import Literal, assert_never

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO
from numpy.typing import ArrayLike

from asdex.decompression import hessian_from_coloring, jacobian_from_coloring
from asdex.modes import _assert_jacobian_mode
from asdex.pattern import ColoredPattern


class VerificationError(AssertionError):
    """Raised when asdex's sparse result does not match JAX's dense reference.

    This indicates that the detected sparsity pattern is missing nonzeros,
    which is a bug — asdex's patterns should always be conservative
    (i.e., contain at least all true nonzeros).
    If you encounter this error,
    please help out asdex's development by reporting this at
    https://github.com/adrhill/asdex/issues.
    """


def check_jacobian_correctness(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    coloring: ColoredPattern,
    *,
    method: Literal["matvec", "dense"] = "matvec",
    num_probes: int = 25,
    seed: int = 0,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    """Verify asdex's sparse Jacobian against a JAX reference at a given input.

    Args:
        f: Function taking an array and returning an array.
        x: Input at which to evaluate the Jacobian.
        coloring: Pre-computed colored pattern from
            :func:`~asdex.jacobian_coloring`.
        method: Verification method.
            ``"matvec"`` uses randomized matrix-vector products,
            which is O(k) in the number of probes.
            ``"dense"`` materializes the full dense Jacobian,
            which is O(n^2).
        num_probes: Number of random probe vectors (only used by ``"matvec"``).
        seed: PRNG seed for reproducibility (only used by ``"matvec"``).
        rtol: Relative tolerance for comparison.
            Defaults to 1e-5 for ``"matvec"`` and 1e-7 for ``"dense"``.
        atol: Absolute tolerance for comparison.
            Defaults to 1e-5 for ``"matvec"`` and 1e-7 for ``"dense"``.

    Raises:
        VerificationError: If the sparse and reference Jacobians disagree.
    """
    if method not in ("matvec", "dense"):
        raise ValueError(f"Unknown method {method!r}. Expected 'matvec' or 'dense'.")

    x = jnp.asarray(x)

    # Derive reference AD mode from the colored pattern
    _assert_jacobian_mode(coloring.mode)
    match coloring.mode:
        case "fwd" | "rev":
            ref_mode = coloring.mode
        case _ as unreachable:
            assert_never(unreachable)  # ty: ignore[type-assertion-failure]

    J_sparse = jacobian_from_coloring(f, coloring)(x)

    match method:
        case "dense":
            jac_fn = jax.jacfwd if ref_mode == "fwd" else jax.jacrev
            J_dense = jac_fn(f)(x)
            _check_allclose(
                J_sparse.todense(), J_dense, "Jacobian", rtol=rtol, atol=atol
            )
        case "matvec":
            _check_jacobian_matvec(
                f,
                x,
                J_sparse,
                ref_mode=ref_mode,
                num_probes=num_probes,
                seed=seed,
                rtol=rtol,
                atol=atol,
            )
        case _ as unreachable:
            assert_never(unreachable)


def check_hessian_correctness(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    coloring: ColoredPattern,
    *,
    method: Literal["matvec", "dense"] = "matvec",
    num_probes: int = 25,
    seed: int = 0,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    """Verify asdex's sparse Hessian against a JAX reference at a given input.

    Args:
        f: Scalar-valued function taking an array.
        x: Input at which to evaluate the Hessian.
        coloring: Pre-computed colored pattern from
            :func:`~asdex.hessian_coloring`.
        method: Verification method.
            ``"matvec"`` uses randomized matrix-vector products,
            which is O(k) in the number of probes.
            ``"dense"`` materializes the full dense Hessian,
            which is O(n^2).
        num_probes: Number of random probe vectors (only used by ``"matvec"``).
        seed: PRNG seed for reproducibility (only used by ``"matvec"``).
        rtol: Relative tolerance for comparison.
            Defaults to 1e-5 for ``"matvec"`` and 1e-7 for ``"dense"``.
        atol: Absolute tolerance for comparison.
            Defaults to 1e-5 for ``"matvec"`` and 1e-7 for ``"dense"``.

    Raises:
        VerificationError: If the sparse and reference Hessians disagree.
    """
    if method not in ("matvec", "dense"):
        raise ValueError(f"Unknown method {method!r}. Expected 'matvec' or 'dense'.")

    x = jnp.asarray(x)

    # Derive reference AD mode from the colored pattern
    match coloring.mode:
        case "fwd_over_rev" | "rev_over_fwd" | "rev_over_rev":
            hessian_mode = coloring.mode
        case "fwd" | "rev":
            raise ValueError(f"Expected a Hessian mode, got {coloring.mode!r}.")
        case _ as unreachable:
            assert_never(unreachable)

    H_sparse = hessian_from_coloring(f, coloring)(x)

    match method:
        case "dense":
            H_dense = _dense_hessian(f, x, hessian_mode)
            _check_allclose(
                H_sparse.todense(), H_dense, "Hessian", rtol=rtol, atol=atol
            )
        case "matvec":
            _check_hessian_matvec(
                f,
                x,
                H_sparse,
                hessian_mode=hessian_mode,
                num_probes=num_probes,
                seed=seed,
                rtol=rtol,
                atol=atol,
            )
        case _ as unreachable:
            assert_never(unreachable)


# Private helpers


def _dense_hessian(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    mode: Literal["fwd_over_rev", "rev_over_fwd", "rev_over_rev"],
) -> jax.Array:
    """Compute a dense Hessian using the specified AD composition."""
    match mode:
        case "fwd_over_rev":
            return jax.jacfwd(jax.grad(f))(x)
        case "rev_over_fwd":
            return jax.jacrev(jax.jacfwd(f))(x)
        case "rev_over_rev":
            return jax.jacrev(jax.grad(f))(x)
        case _ as unreachable:
            assert_never(unreachable)


def _check_jacobian_matvec(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    J_sparse: BCOO,
    *,
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

    out_shape = jax.eval_shape(f, x).shape
    m = int(np.prod(out_shape))
    n = x.size

    for i in range(num_probes):
        match ref_mode:
            case "fwd":
                v = jax.random.normal(keys[i], shape=(n,))
                sparse_result = (J_sparse @ v).ravel()
                _, ref_result = jax.jvp(f, (x,), (v.reshape(x.shape),))
                ref_result = jnp.asarray(ref_result).ravel()
            case "rev":
                v = jax.random.normal(keys[i], shape=(m,))
                sparse_result = (v @ J_sparse).ravel()
                _, vjp_fn = jax.vjp(f, x)
                (ref_result,) = vjp_fn(v.reshape(out_shape))
                ref_result = jnp.asarray(ref_result).ravel()
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
    x: jax.Array,
    H_sparse: BCOO,
    *,
    hessian_mode: Literal["fwd_over_rev", "rev_over_fwd", "rev_over_rev"],
    num_probes: int,
    seed: int,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    """Verify a sparse Hessian via randomized H @ v products."""
    rtol = rtol if rtol is not None else 1e-5
    atol = atol if atol is not None else 1e-5
    n = x.size
    key = jax.random.key(seed)
    keys = jax.random.split(key, num_probes)

    match hessian_mode:
        case "fwd_over_rev":

            def hvp(v: jax.Array) -> jax.Array:
                _, result = jax.jvp(jax.grad(f), (x,), (v.reshape(x.shape),))
                return jnp.asarray(result).ravel()

        case "rev_over_fwd":

            def hvp(v: jax.Array) -> jax.Array:
                result = jax.grad(lambda p: jax.jvp(f, (p,), (v.reshape(x.shape),))[1])(
                    x
                )
                return jnp.asarray(result).ravel()

        case "rev_over_rev":

            def hvp(v: jax.Array) -> jax.Array:
                result = jax.grad(
                    lambda y: jnp.vdot(jax.grad(f)(y), v.reshape(x.shape))
                )(x)
                return jnp.asarray(result).ravel()

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
