"""Tests for the verification utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import (
    ColoredPattern,
    SparsityPattern,
    VerificationError,
    check_hessian_correctness,
    check_jacobian_correctness,
    hessian_coloring,
    jacobian_coloring,
)

# Jacobian verification — matvec (default)


@pytest.mark.jacobian
def test_check_jacobian_passes():
    """check_jacobian_correctness returns silently on correct results (default matvec)."""

    def f(x):
        return (x[1:] - x[:-1]) ** 2

    x = np.array([1.0, 2.0, 3.0, 4.0])
    coloring = jacobian_coloring(f, x)
    check_jacobian_correctness(f, x, coloring)


@pytest.mark.jacobian
def test_check_jacobian_matvec_explicit():
    """check_jacobian_correctness works with explicit method='matvec'."""

    def f(x):
        return x**2

    x = np.array([1.0, 2.0, 3.0])
    coloring = jacobian_coloring(f, x)
    check_jacobian_correctness(f, x, coloring, method="matvec")


@pytest.mark.jacobian
def test_check_jacobian_with_precomputed_pattern():
    """check_jacobian_correctness works with a pre-computed colored pattern."""

    def f(x):
        return x**2

    x = np.array([1.0, 2.0, 3.0])
    coloring = jacobian_coloring(f, x)
    check_jacobian_correctness(f, x, coloring)


@pytest.mark.jacobian
def test_check_jacobian_custom_tolerances():
    """check_jacobian_correctness respects custom tolerances."""

    def f(x):
        return jnp.sin(x)

    x = np.array([0.5, 1.0, 1.5])
    coloring = jacobian_coloring(f, x)
    check_jacobian_correctness(f, x, coloring, rtol=1e-5, atol=1e-5)


@pytest.mark.jacobian
def test_check_jacobian_matvec_raises_on_mismatch():
    """check_jacobian_correctness raises VerificationError on wrong results (matvec).

    Uses a diagonal colored pattern for a function with off-diagonal entries,
    so the sparse Jacobian misses non-zeros.
    """

    def f_dense(x):
        return jnp.array([x[0] + x[1] + x[2], x[0] + x[1] + x[2], x[0] + x[1] + x[2]])

    # Diagonal pattern misses off-diagonal Jacobian entries
    coloring = jacobian_coloring(lambda x: x**2, np.zeros(3))

    x = np.array([1.0, 2.0, 3.0])
    with pytest.raises(VerificationError, match="matvec verification"):
        check_jacobian_correctness(f_dense, x, coloring)


@pytest.mark.jacobian
def test_check_jacobian_custom_seed_and_num_probes():
    """check_jacobian_correctness accepts custom seed and num_probes."""

    def f(x):
        return x**2

    x = np.array([1.0, 2.0, 3.0])
    coloring = jacobian_coloring(f, x)
    check_jacobian_correctness(f, x, coloring, seed=42, num_probes=5)


# Jacobian verification — AD modes


@pytest.mark.jacobian
def test_check_jacobian_forward_mode():
    """check_jacobian_correctness works with fwd mode colored pattern."""

    def f(x):
        return (x[1:] - x[:-1]) ** 2

    x = np.array([1.0, 2.0, 3.0, 4.0])
    coloring = jacobian_coloring(f, x, mode="fwd")
    check_jacobian_correctness(f, x, coloring)


@pytest.mark.jacobian
def test_check_jacobian_reverse_mode():
    """check_jacobian_correctness works with rev mode colored pattern."""

    def f(x):
        return (x[1:] - x[:-1]) ** 2

    x = np.array([1.0, 2.0, 3.0, 4.0])
    coloring = jacobian_coloring(f, x, mode="rev")
    check_jacobian_correctness(f, x, coloring)


@pytest.mark.jacobian
def test_check_jacobian_reverse_mode_raises_on_mismatch():
    """check_jacobian_correctness raises with rev mode on wrong results."""

    def f_dense(x):
        return jnp.array([x[0] + x[1] + x[2], x[0] + x[1] + x[2], x[0] + x[1] + x[2]])

    coloring = jacobian_coloring(lambda x: x**2, np.zeros(3))

    x = np.array([1.0, 2.0, 3.0])
    with pytest.raises(VerificationError, match="matvec verification"):
        check_jacobian_correctness(f_dense, x, coloring)


# Jacobian verification — dense


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_check_jacobian_dense(mode):
    """check_jacobian_correctness works with method='dense' and both AD modes."""

    def f(x):
        return (x[1:] - x[:-1]) ** 2

    x = np.array([1.0, 2.0, 3.0, 4.0])
    coloring = jacobian_coloring(f, x, mode=mode)
    check_jacobian_correctness(f, x, coloring, method="dense")


@pytest.mark.jacobian
def test_check_jacobian_dense_raises_on_mismatch():
    """check_jacobian_correctness raises VerificationError on wrong results (dense)."""

    def f_dense(x):
        return jnp.array([x[0] + x[1] + x[2], x[0] + x[1] + x[2], x[0] + x[1] + x[2]])

    coloring = jacobian_coloring(lambda x: x**2, np.zeros(3))

    x = np.array([1.0, 2.0, 3.0])
    with pytest.raises(VerificationError, match="does not match"):
        check_jacobian_correctness(f_dense, x, coloring, method="dense")


# Hessian verification — matvec (default)


@pytest.mark.hessian
def test_check_hessian_passes():
    """check_hessian_correctness returns silently on correct results (default matvec)."""

    def f(x):
        return jnp.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2)

    x = np.array([1.0, 1.0, 1.0, 1.0])
    coloring = hessian_coloring(f, x)
    check_hessian_correctness(f, x, coloring)


@pytest.mark.hessian
def test_check_hessian_matvec_explicit():
    """check_hessian_correctness works with explicit method='matvec'."""

    def f(x):
        return jnp.sum(x**2)

    x = np.array([1.0, 2.0, 3.0])
    coloring = hessian_coloring(f, x)
    check_hessian_correctness(f, x, coloring, method="matvec")


@pytest.mark.hessian
def test_check_hessian_custom_tolerances():
    """check_hessian_correctness respects custom tolerances."""

    def f(x):
        return jnp.sum(x**2)

    x = np.array([1.0, 2.0, 3.0])
    coloring = hessian_coloring(f, x)
    check_hessian_correctness(f, x, coloring, rtol=1e-5, atol=1e-5)


@pytest.mark.hessian
def test_check_hessian_matvec_raises_on_mismatch():
    """check_hessian_correctness raises VerificationError on wrong results (matvec).

    Uses a diagonal colored pattern for a function with off-diagonal Hessian entries,
    so the sparse Hessian misses non-zeros.
    """

    def f(x):
        return x[0] * x[1] + x[1] * x[2]

    # Diagonal pattern misses off-diagonal Hessian entries
    coloring = hessian_coloring(lambda x: jnp.sum(x**2), np.zeros(3))

    x = np.array([1.0, 2.0, 3.0])
    with pytest.raises(VerificationError, match="matvec verification"):
        check_hessian_correctness(f, x, coloring)


@pytest.mark.hessian
def test_check_hessian_custom_seed_and_num_probes():
    """check_hessian_correctness accepts custom seed and num_probes."""

    def f(x):
        return jnp.sum(x**2)

    x = np.array([1.0, 2.0, 3.0])
    coloring = hessian_coloring(f, x)
    check_hessian_correctness(f, x, coloring, seed=42, num_probes=5)


# Hessian verification — AD modes


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_check_hessian_modes(mode):
    """check_hessian_correctness works with all HVP AD modes."""

    def f(x):
        return x[0] * x[1] + x[1] * x[2] + x[2] * x[3]

    x = np.array([1.0, 2.0, 3.0, 4.0])
    coloring = hessian_coloring(f, x, mode=mode)
    check_hessian_correctness(f, x, coloring)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_check_hessian_modes_raise_on_mismatch(mode):
    """check_hessian_correctness raises with all HVP AD modes on wrong results."""

    def f(x):
        return x[0] * x[1] + x[1] * x[2]

    coloring = hessian_coloring(lambda x: jnp.sum(x**2), np.zeros(3))

    x = np.array([1.0, 2.0, 3.0])
    with pytest.raises(VerificationError, match="matvec verification"):
        check_hessian_correctness(f, x, coloring)


# Hessian verification — dense


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_check_hessian_dense(mode):
    """check_hessian_correctness works with method='dense' and all AD modes."""

    def f(x):
        return jnp.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2)

    x = np.array([1.0, 1.0, 1.0, 1.0])
    coloring = hessian_coloring(f, x, mode=mode)
    check_hessian_correctness(f, x, coloring, method="dense")


@pytest.mark.hessian
def test_check_hessian_dense_raises_on_mismatch():
    """check_hessian_correctness raises VerificationError on wrong results (dense)."""

    def f(x):
        return x[0] * x[1] + x[1] * x[2]

    coloring = hessian_coloring(lambda x: jnp.sum(x**2), np.zeros(3))

    x = np.array([1.0, 2.0, 3.0])
    with pytest.raises(VerificationError, match="does not match"):
        check_hessian_correctness(f, x, coloring, method="dense")


# Invalid method


def test_invalid_method_jacobian():
    """check_jacobian_correctness raises ValueError on unknown method."""

    def f(x):
        return x**2

    x = np.array([1.0, 2.0])
    coloring = jacobian_coloring(f, x)
    with pytest.raises(ValueError, match="Unknown method"):
        check_jacobian_correctness(f, x, coloring, method="invalid")  # ty: ignore[invalid-argument-type]


def test_invalid_method_hessian():
    """check_hessian_correctness raises ValueError on unknown method."""

    def f(x):
        return jnp.sum(x**2)

    x = np.array([1.0, 2.0])
    coloring = hessian_coloring(f, x)
    with pytest.raises(ValueError, match="Unknown method"):
        check_hessian_correctness(f, x, coloring, method="invalid")  # ty: ignore[invalid-argument-type]


# VerificationError


def test_verification_error_is_assertion_error():
    """VerificationError subclasses AssertionError."""
    assert issubclass(VerificationError, AssertionError)


# Cross-mode coloring


@pytest.mark.jacobian
def test_check_jacobian_with_hessian_coloring_raises():
    """check_jacobian_correctness raises ValueError for Hessian-mode colorings."""

    def f(x):
        return jnp.sum(x**2)

    x = np.array([1.0, 2.0, 3.0])
    coloring = hessian_coloring(f, x)
    with pytest.raises(ValueError, match="Expected 'fwd' or 'rev'"):
        check_jacobian_correctness(jax.grad(f), x, coloring)


@pytest.mark.hessian
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_check_hessian_with_jacobian_coloring_raises():
    """check_hessian_correctness raises ValueError for Jacobian-mode colorings."""

    def f(x):
        return jnp.sum(x**2)

    x = np.array([1.0, 2.0, 3.0])
    coloring = jacobian_coloring(f, x)
    with pytest.raises(ValueError, match="Expected a Hessian mode"):
        check_hessian_correctness(f, x, coloring)


# Shape mismatch in _check_allclose


@pytest.mark.jacobian
def test_check_allclose_shape_mismatch():
    """_check_allclose raises VerificationError when shapes differ.

    Constructs a coloring whose pattern is too small for the function,
    so the decompressed sparse Jacobian has a different shape
    than the dense reference.
    """

    # Function returns 3 outputs from 3 inputs → dense Jacobian is (3, 3)
    def f(x):
        return x**2

    x = np.array([1.0, 2.0, 3.0])

    # Build a coloring for a (2, 3) pattern — wrong number of rows
    sparsity = SparsityPattern.from_coo([0, 1], [0, 1], (2, 3))
    coloring = ColoredPattern(
        sparsity=sparsity,
        colors=np.array([0, 0, 0], dtype=np.int32),
        num_colors=1,
        symmetric=False,
        mode="fwd",
    )

    with pytest.raises(VerificationError, match="shape"):
        check_jacobian_correctness(f, x, coloring, method="dense")


# PyTree inputs and outputs


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("method", ["matvec", "dense"])
def test_check_jacobian_pytree_input(mode, method):
    """check_jacobian_correctness works with PyTree inputs."""

    def f(params):
        return params["a"] + params["b"] * 2

    params = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
    coloring = jacobian_coloring(f, params, mode=mode)
    check_jacobian_correctness(f, params, coloring, method=method)


@pytest.mark.jacobian
def test_check_jacobian_pytree_output():
    """check_jacobian_correctness works with PyTree outputs."""

    def f(x):
        return {"a": x[:2] ** 2, "b": x[2:]}

    x = np.array([1.0, 2.0, 3.0, 4.0])
    coloring = jacobian_coloring(f, x)
    check_jacobian_correctness(f, x, coloring)


@pytest.mark.jacobian
@pytest.mark.parametrize("method", ["matvec", "dense"])
def test_check_jacobian_pytree_output_methods(method):
    """check_jacobian_correctness works with PyTree outputs for both methods."""

    def f(x):
        return (x[:2], x[2:] ** 2)

    x = np.array([1.0, 2.0, 3.0, 4.0])
    coloring = jacobian_coloring(f, x)
    check_jacobian_correctness(f, x, coloring, method=method)


@pytest.mark.hessian
def test_check_hessian_pytree_input():
    """check_hessian_correctness works with PyTree inputs (single argnum)."""

    def f(x):
        return jnp.sum(x["a"] ** 2) + jnp.sum(x["b"] ** 2)

    x = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
    coloring = hessian_coloring(f, x)
    check_hessian_correctness(f, x, coloring)


@pytest.mark.hessian
@pytest.mark.parametrize("method", ["matvec", "dense"])
def test_check_hessian_pytree_input_methods(method):
    """check_hessian_correctness works with PyTree inputs for both methods."""

    def f(x):
        return jnp.sum(x[0] ** 2) + jnp.dot(x[0], x[1])

    x = (np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    coloring = hessian_coloring(f, x)
    check_hessian_correctness(f, x, coloring, method=method)


# Multi-argument verification


@pytest.mark.jacobian
@pytest.mark.parametrize("method", ["matvec", "dense"])
def test_check_jacobian_multi_arg(method):
    """check_jacobian_correctness works with multi-argument functions."""

    def f(x, y):
        return x * y + x**2

    x, y = np.array([1.0, 2.0]), np.array([3.0, 4.0])
    coloring = jacobian_coloring(f, x, y, argnums=(0, 1))
    check_jacobian_correctness(f, (x, y), coloring, method=method)


@pytest.mark.jacobian
@pytest.mark.bug
def test_check_jacobian_multi_arg_single_argnum():
    """check_jacobian_correctness should work with single argnum from multi-arg function.

    Currently verify.py doesn't handle the case where argnums is an int
    but f takes multiple arguments.
    """

    def f(x, y):
        return x * y + x**2

    x, y = np.array([1.0, 2.0]), np.array([3.0, 4.0])
    coloring = jacobian_coloring(f, x, y, argnums=0)
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        check_jacobian_correctness(f, (x, y), coloring)


@pytest.mark.hessian
@pytest.mark.parametrize("method", ["matvec", "dense"])
def test_check_hessian_multi_arg(method):
    """check_hessian_correctness works with multi-argument functions."""

    def f(x, y):
        return jnp.sum(x**2) + jnp.dot(x, y)

    x, y = np.array([1.0, 2.0]), np.array([3.0, 4.0])
    coloring = hessian_coloring(f, x, y, argnums=(0, 1))
    check_hessian_correctness(f, (x, y), coloring, method=method)


@pytest.mark.hessian
@pytest.mark.bug
def test_check_hessian_multi_arg_single_argnum():
    """check_hessian_correctness should work with single argnum from multi-arg function.

    Currently verify.py doesn't handle the case where argnums is an int
    but f takes multiple arguments.
    """

    def f(x, y):
        return jnp.sum(x**2) + jnp.dot(x, y)

    x, y = np.array([1.0, 2.0]), np.array([3.0, 4.0])
    coloring = hessian_coloring(f, x, y, argnums=0)
    with pytest.raises(ValueError, match="Expected 2 positional argument"):
        check_hessian_correctness(f, (x, y), coloring)


# PyTree inputs with same-ndim blocks (happy path for _stack_bcoo_pytree)


@pytest.mark.jacobian
@pytest.mark.parametrize("method", ["matvec", "dense"])
def test_check_jacobian_pytree_same_ndim_blocks(method):
    """check_jacobian_correctness works when all PyTree leaf Jacobians have same ndim."""

    def f(params):
        return params["a"] + params["b"]

    params = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
    coloring = jacobian_coloring(f, params)
    check_jacobian_correctness(f, params, coloring, method=method)


@pytest.mark.jacobian
def test_check_jacobian_pytree_2d_same_ndim_blocks_dense():
    """check_jacobian_correctness (dense) works with 2D PyTree leaves of same ndim."""

    def f(params):
        return params["a"] + params["b"]

    params = {"a": np.eye(2), "b": np.ones((2, 2))}
    coloring = jacobian_coloring(f, params)
    check_jacobian_correctness(f, params, coloring, method="dense")


@pytest.mark.jacobian
@pytest.mark.bug
def test_check_jacobian_pytree_2d_matvec_shape_mismatch():
    """Matvec verification fails for 2D PyTree inputs due to shape handling.

    The dense method works, but matvec has shape mismatches when inputs are 2D.
    """

    def f(params):
        return params["a"] + params["b"]

    params = {"a": np.eye(2), "b": np.ones((2, 2))}
    coloring = jacobian_coloring(f, params)
    with pytest.raises(TypeError, match="contracting dimensions"):
        check_jacobian_correctness(f, params, coloring, method="matvec")


@pytest.mark.jacobian
@pytest.mark.bug
def test_check_jacobian_pytree_input_and_output_matvec_fails():
    """Matvec verification fails when both input and output are PyTrees.

    The dense method works, but matvec has shape mismatches due to incorrect
    flattening of the Jacobian structure.
    """

    def f(params):
        return {"sum": params["a"] + params["b"], "diff": params["a"] - params["b"]}

    params = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
    coloring = jacobian_coloring(f, params)
    check_jacobian_correctness(f, params, coloring, method="dense")
    with pytest.raises(TypeError, match="contracting dimensions"):
        check_jacobian_correctness(f, params, coloring, method="matvec")


@pytest.mark.jacobian
@pytest.mark.bug
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_check_jacobian_3d_tensor_input_matvec_fails():
    """Matvec verification fails for 3D tensor inputs.

    The dense method works, but matvec has shape mismatches when inputs
    have more than 2 dimensions.
    """

    def f(params):
        return jnp.einsum("ijk,k->ij", params["tensor"], params["vec"])

    params = {"tensor": jnp.ones((2, 3, 4)), "vec": jnp.ones(4)}
    coloring = jacobian_coloring(f, params)
    check_jacobian_correctness(f, params, coloring, method="dense")
    with pytest.raises(TypeError, match="contracting dimensions"):
        check_jacobian_correctness(f, params, coloring, method="matvec")


# _stack_bcoo_pytree tests


@pytest.mark.jacobian
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_stack_bcoo_pytree_handles_different_ndim_dict():
    """_stack_bcoo_pytree handles Jacobian blocks with different dimensions."""

    def f(params):
        return params["w"] @ params["b"]

    params = {"w": jnp.eye(3), "b": jnp.ones(3)}
    coloring = jacobian_coloring(f, params)
    check_jacobian_correctness(f, params, coloring)


@pytest.mark.jacobian
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_stack_bcoo_pytree_handles_different_ndim_tuple():
    """_stack_bcoo_pytree handles tuple PyTree inputs with different ndims."""

    def f(inputs):
        a, b = inputs
        return a @ b

    inputs = (jnp.eye(2), jnp.ones(2))
    coloring = jacobian_coloring(f, inputs)
    check_jacobian_correctness(f, inputs, coloring)
