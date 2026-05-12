"""Tests for PyTree input and output structures.

Verifies that asdex handles complex nested PyTrees (dicts, lists, tuples,
namedtuples) the same way as JAX.
"""

import warnings
from collections import namedtuple

import jax
import jax.numpy as jnp
import pytest

import asdex

warnings.filterwarnings("ignore", category=asdex.DenseColoringWarning)


# Nested dict inputs


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_nested_dict_input(mode, output_format, assert_trees_allclose):
    """Nested dict inputs (dict of dicts) match jax.jacobian."""

    def f(params):
        return params["layer1"]["w"] @ params["layer2"]["w"].T

    params = {
        "layer1": {"w": jnp.array([[1.0, 2.0], [3.0, 4.0]])},
        "layer2": {"w": jnp.array([[5.0, 6.0], [7.0, 8.0]])},
    }
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_triple_nested_dict(mode, output_format, assert_trees_allclose):
    """Triple-nested dict input matches jax.jacobian."""
    if output_format == "bcoo":
        pytest.xfail("BCOO Jacobians lose PyTree structure for nested dict inputs")

    def f(params):
        return params["net"]["layer"]["w"] @ jnp.ones(2)

    params = {"net": {"layer": {"w": jnp.eye(3, 2)}}}
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_nested_dict_input(mode, output_format, assert_trees_allclose):
    """Hessian with nested dict input matches jax.hessian."""
    if output_format == "bcoo":
        pytest.xfail("BCOO Hessians lose PyTree structure for nested dict inputs")

    def f(params):
        w = params["layer"]["w"]
        return jnp.sum(w**2)

    params = {"layer": {"w": jnp.array([1.0, 2.0, 3.0])}}
    H = asdex.hessian(f, params, mode=mode, output_format=output_format)(params)
    H_jax = jax.hessian(f)(params)
    assert_trees_allclose(H, H_jax, atol=1e-6)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_triple_nested_dict(mode, output_format, assert_trees_allclose):
    """Hessian with triple-nested dict input matches jax.hessian."""
    if output_format == "bcoo":
        pytest.xfail("BCOO Hessians lose PyTree structure for nested dict inputs")

    def f(params):
        w = params["net"]["layer"]["w"]
        return jnp.sum(w**2)

    params = {"net": {"layer": {"w": jnp.array([1.0, 2.0, 3.0])}}}
    H = asdex.hessian(f, params, mode=mode, output_format=output_format)(params)
    H_jax = jax.hessian(f)(params)
    assert_trees_allclose(H, H_jax, atol=1e-6)


# Nested dict outputs


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_nested_dict_output(mode, output_format, assert_trees_allclose):
    """Nested dict outputs match jax.jacobian."""

    def f(x):
        return {
            "stats": {"mean": jnp.mean(x), "sum": jnp.sum(x)},
            "scaled": x * 2,
        }

    x = jnp.array([1.0, 2.0, 3.0])
    J = asdex.jacobian(f, x, mode=mode, output_format=output_format)(x)
    J_jax = jax.jacobian(f)(x)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_triple_nested_dict_output(mode, output_format, assert_trees_allclose):
    """Triple-nested dict output matches jax.jacobian."""
    if output_format == "bcoo":
        pytest.xfail("BCOO Jacobians lose PyTree structure for deeply nested outputs")

    def f(x):
        return {"level1": {"level2": {"y": x**2}}}

    x = jnp.array([1.0, 2.0, 3.0])
    J = asdex.jacobian(f, x, mode=mode, output_format=output_format)(x)
    J_jax = jax.jacobian(f)(x)
    assert_trees_allclose(J, J_jax)


# List inputs and outputs


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_list_of_arrays_input(mode, output_format, assert_trees_allclose):
    """List of arrays as input matches jax.jacobian."""

    def f(params):
        return params[0] + params[1] * 2 + params[2] * 3

    params = [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]), jnp.array([5.0, 6.0])]
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_list_output(mode, output_format, assert_trees_allclose):
    """List as output matches jax.jacobian."""

    def f(x):
        return [x[:2], x[1:] * 2]

    x = jnp.array([1.0, 2.0, 3.0])
    J = asdex.jacobian(f, x, mode=mode, output_format=output_format)(x)
    J_jax = jax.jacobian(f)(x)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_list_input_list_output(mode, output_format, assert_trees_allclose):
    """List input and list output matches jax.jacobian."""

    def f(params):
        return [params[0] + params[1], params[0] * params[1]]

    params = [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])]
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


# Tuple outputs


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_tuple_output(mode, output_format, assert_trees_allclose):
    """Tuple as output matches jax.jacobian."""

    def f(x):
        return (x[:2], x[1:] * 2, x**2)

    x = jnp.array([1.0, 2.0, 3.0])
    J = asdex.jacobian(f, x, mode=mode, output_format=output_format)(x)
    J_jax = jax.jacobian(f)(x)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_nested_tuple_output(mode, output_format, assert_trees_allclose):
    """Nested tuple as output matches jax.jacobian."""

    def f(x):
        return ((x[:2], x[1:]), (x * 2,))

    x = jnp.array([1.0, 2.0, 3.0])
    J = asdex.jacobian(f, x, mode=mode, output_format=output_format)(x)
    J_jax = jax.jacobian(f)(x)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_tuple_of_dicts_output(mode, output_format, assert_trees_allclose):
    """Tuple of dicts as output matches jax.jacobian."""

    def f(x):
        return ({"a": x[:2]}, {"b": x[1:], "c": x * 2})

    x = jnp.array([1.0, 2.0, 3.0])
    J = asdex.jacobian(f, x, mode=mode, output_format=output_format)(x)
    J_jax = jax.jacobian(f)(x)
    assert_trees_allclose(J, J_jax)


# Mixed container types


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_dict_containing_list(mode, output_format, assert_trees_allclose):
    """Dict containing list as input matches jax.jacobian."""

    def f(params):
        return params["weights"][0] + params["weights"][1] * 2

    params = {"weights": [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])]}
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_dict_containing_tuple(mode, output_format, assert_trees_allclose):
    """Dict containing tuple as input matches jax.jacobian."""

    def f(params):
        w, b = params["layer"]
        return w @ jnp.ones(2) + b

    params = {"layer": (jnp.eye(3, 2), jnp.zeros(3))}
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_mixed_depths_in_dict(mode, output_format, assert_trees_allclose):
    """Dict with mixed nesting depths matches jax.jacobian."""

    def f(params):
        return params["shallow"] + params["deep"]["inner"]["w"]

    params = {
        "shallow": jnp.array([1.0, 2.0]),
        "deep": {"inner": {"w": jnp.array([3.0, 4.0])}},
    }
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_heterogeneous_output_pytree(
    mode, output_format, assert_trees_allclose
):
    """Output with mixed dict, tuple, list matches jax.jacobian."""

    def f(x):
        return {"a": x[:2]}, (x[1:],), [x * 2]

    x = jnp.array([1.0, 2.0, 3.0])
    J = asdex.jacobian(f, x, mode=mode, output_format=output_format)(x)
    J_jax = jax.jacobian(f)(x)
    assert_trees_allclose(J, J_jax)


# Namedtuple inputs


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_namedtuple_input(mode, output_format, assert_trees_allclose):
    """Namedtuple input matches jax.jacobian."""
    Params = namedtuple("Params", ["w", "b"])

    def f(params):
        return params.w @ jnp.ones(2) + params.b

    params = Params(w=jnp.eye(3, 2), b=jnp.zeros(3))
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_namedtuple_input(mode, output_format, assert_trees_allclose):
    """Hessian with namedtuple input matches jax.hessian."""
    Params = namedtuple("Params", ["a", "b"])

    def f(params):
        return jnp.sum(params.a**2) + jnp.dot(params.a, params.b)

    params = Params(a=jnp.array([1.0, 2.0]), b=jnp.array([3.0, 4.0]))
    H = asdex.hessian(f, params, mode=mode, output_format=output_format)(params)
    H_jax = jax.hessian(f)(params)
    assert_trees_allclose(H, H_jax, atol=1e-6)


# Mixed PyTree input and output


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_mixed_pytree_input_output(mode, output_format, assert_trees_allclose):
    """Both input and output as complex PyTrees match jax.jacobian."""

    def f(params):
        w, b = params["w"], params["b"]
        return {"y": w @ jnp.ones(w.shape[1]) + b, "norm": jnp.sum(w**2)}

    params = {"w": jnp.eye(3, 2), "b": jnp.zeros(3)}
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_nested_input_nested_output(
    mode, output_format, assert_trees_allclose
):
    """Both nested input and nested output match jax.jacobian."""
    if output_format == "bcoo":
        pytest.xfail("BCOO Jacobians lose PyTree structure for nested input + output")

    def f(params):
        w = params["net"]["layer"]["w"]
        return {"out": {"pred": w @ jnp.ones(2)}}

    params = {"net": {"layer": {"w": jnp.eye(3, 2)}}}
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


# Scalar and single-element leaves


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_scalar_leaves_in_pytree(mode, output_format, assert_trees_allclose):
    """PyTree with scalar leaves matches jax.jacobian."""

    def f(params):
        return params["a"] * params["scale"] + params["offset"]

    params = {
        "a": jnp.array([1.0, 2.0]),
        "scale": jnp.array(2.0),
        "offset": jnp.array(1.0),
    }
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_single_element_arrays(mode, output_format, assert_trees_allclose):
    """PyTree with single-element arrays matches jax.jacobian."""

    def f(params):
        return params["vec"] * params["scalar"][0]

    params = {"vec": jnp.array([1.0, 2.0, 3.0]), "scalar": jnp.array([2.0])}
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_single_leaf_pytree(mode, output_format, assert_trees_allclose):
    """Single-leaf PyTree behaves like the leaf itself."""
    if output_format == "bcoo":
        pytest.xfail("BCOO Jacobians lose PyTree structure for single-leaf dict")

    def f(params):
        return params["w"] ** 2

    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_all_scalar_leaves(mode, output_format, assert_trees_allclose):
    """PyTree with all scalar leaves matches jax.jacobian."""

    def f(params):
        return jnp.array([params["a"] + params["b"], params["a"] * params["b"]])

    params = {"a": jnp.array(2.0), "b": jnp.array(3.0)}
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_all_scalar_leaves(mode, output_format, assert_trees_allclose):
    """Hessian with all scalar leaves in PyTree matches jax.hessian."""

    def f(params):
        return params["a"] ** 2 + params["a"] * params["b"] + params["b"] ** 2

    params = {"a": jnp.array(2.0), "b": jnp.array(3.0)}
    H = asdex.hessian(f, params, mode=mode, output_format=output_format)(params)
    H_jax = jax.hessian(f)(params)
    assert_trees_allclose(H, H_jax, atol=1e-6)


# Multi-dimensional array leaves


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_2d_arrays_in_pytree(mode, output_format, assert_trees_allclose):
    """PyTree with 2D array leaves matches jax.jacobian."""

    def f(params):
        return params["W"] @ params["x"]

    params = {"W": jnp.eye(3, 2), "x": jnp.array([1.0, 2.0])}
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_3d_arrays_in_pytree(mode, output_format, assert_trees_allclose):
    """PyTree with 3D array leaves matches jax.jacobian."""

    def f(params):
        return jnp.sum(params["tensor"], axis=(0, 1)) + params["bias"]

    params = {"tensor": jnp.ones((2, 3, 4)), "bias": jnp.zeros(4)}
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_2d_array_input(mode, output_format, assert_trees_allclose):
    """Hessian with 2D array in PyTree matches jax.hessian."""
    if output_format == "bcoo":
        pytest.xfail("BCOO Hessians lose PyTree structure for dict with 2D array")

    def f(params):
        W = params["W"]
        return jnp.sum(W**2) + jnp.sum(W)

    params = {"W": jnp.array([[1.0, 2.0], [3.0, 4.0]])}
    H = asdex.hessian(f, params, mode=mode, output_format=output_format)(params)
    H_jax = jax.hessian(f)(params)
    assert_trees_allclose(H, H_jax, atol=1e-6)


# Empty and edge cases


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_empty_array_leaf(mode, output_format, assert_trees_allclose):
    """PyTree with empty array leaf matches jax.jacobian."""

    def f(params):
        return params["w"] ** 2

    params = {"w": jnp.array([1.0, 2.0]), "empty": jnp.array([])}
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_empty_array_leaf(mode, output_format, assert_trees_allclose):
    """Hessian with empty array leaf in PyTree matches jax.hessian."""

    def f(params):
        return jnp.sum(params["w"] ** 2)

    params = {"w": jnp.array([1.0, 2.0]), "empty": jnp.array([])}
    H = asdex.hessian(f, params, mode=mode, output_format=output_format)(params)
    H_jax = jax.hessian(f)(params)
    assert_trees_allclose(H, H_jax, atol=1e-6)


# Sparsity pattern shape consistency


@pytest.mark.jacobian
def test_jacobian_sparsity_shape_matches_jacobian():
    """jacobian_sparsity shape matches flattened jacobian dimensions."""

    def f(params):
        return params["w"] @ params["x"]

    params = {"w": jnp.eye(2, 3), "x": jnp.array([1.0, 2.0, 3.0])}
    pattern = asdex.jacobian_sparsity(f, params)

    total_in = sum(leaf.size for leaf in jax.tree.leaves(params))
    expected_shape = (2, total_in)

    assert pattern.shape == expected_shape


@pytest.mark.hessian
def test_hessian_sparsity_shape_matches_hessian():
    """hessian_sparsity shape matches flattened hessian dimensions."""

    def f(params):
        return jnp.sum(params["a"] ** 2) + jnp.sum(params["b"] ** 2)

    params = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
    pattern = asdex.hessian_sparsity(f, params)

    total_in = sum(leaf.size for leaf in jax.tree.leaves(params))
    expected_shape = (total_in, total_in)

    assert pattern.shape == expected_shape


# Output format consistency


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_jacobian_dense_bcoo_same_structure(mode, assert_trees_allclose):
    """Dense and BCOO outputs have identical logical structure."""

    def f(params):
        return {"y": params["w"] @ params["x"], "z": params["b"]}

    params = {"w": jnp.eye(2, 3), "x": jnp.array([1.0, 2.0, 3.0]), "b": jnp.zeros(2)}
    J_dense = asdex.jacobian(f, params, mode=mode, output_format="dense")(params)
    J_bcoo = asdex.jacobian(f, params, mode=mode, output_format="bcoo")(params)
    assert_trees_allclose(J_bcoo, J_dense)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_hessian_dense_bcoo_same_structure(mode, assert_trees_allclose):
    """Hessian dense and BCOO outputs have identical logical structure."""

    def f(params):
        return jnp.sum(params["a"] ** 2) + jnp.dot(params["a"], params["b"])

    params = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
    H_dense = asdex.hessian(f, params, mode=mode, output_format="dense")(params)
    H_bcoo = asdex.hessian(f, params, mode=mode, output_format="bcoo")(params)
    assert_trees_allclose(H_bcoo, H_dense, atol=1e-6)
