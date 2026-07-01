"""Forward-evaluation counts for the one-shot Jacobian and Hessian APIs.

These pin how many times the user function ``f`` is invoked per call.
On a non-empty pattern the primal value rides the AD forward pass for free in
every mode, so a value-free API and its ``value_and_*`` counterpart each invoke
``f`` exactly once.
The Hessian's ``rev_over_fwd`` lifts the value out of the vmapped HVPs as the
``jax.grad`` aux (each inner ``jax.jvp`` already evaluates ``f``) rather than
paying a dedicated ``f`` call.
The only place a value-free call saves an ``f`` evaluation is the empty-pattern
short-circuit, which has no forward pass to ride and is exercised in the
compression tests.
"""

import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from asdex import hessian, jacobian, value_and_hessian, value_and_jacobian


def _counting(f):
    """Wrap ``f`` with a call counter, returning ``(wrapped, counter)``.

    The counter is reset by the caller after construction so that only
    call-time invocations are counted, not the detection/coloring trace.
    """
    counter = {"n": 0}

    def wrapped(*args, **kwargs):
        counter["n"] += 1
        return f(*args, **kwargs)

    return wrapped, counter


def _jac_f(x):
    """Bidiagonal Jacobian: output i depends on x[i] and x[i+1]."""
    return (x[1:] - x[:-1]) ** 2


def _hess_f(x):
    """Tridiagonal Hessian from a chained quadratic."""
    return jnp.sum((x[1:] - x[:-1]) ** 2) + jnp.sum(x**3)


# Jacobian: the primal value always rides the forward pass for free


@pytest.mark.jacobian
def test_jacobian_value_free_matches_value_and_call_count(jacobian_mode):
    """Value-free and value-returning Jacobians both invoke ``f`` exactly once.

    In both fwd and rev the primal value is a byproduct of the forward pass,
    so returning it costs no extra ``f`` call.
    """
    x = jnp.arange(1.0, 8.0)

    f_free, c_free = _counting(_jac_f)
    fn_free = jacobian(f_free, x, mode=jacobian_mode)
    c_free["n"] = 0
    fn_free(x)

    f_val, c_val = _counting(_jac_f)
    fn_val = value_and_jacobian(f_val, x, mode=jacobian_mode)
    c_val["n"] = 0
    value, _ = fn_val(x)

    assert c_free["n"] == 1
    assert c_val["n"] == 1
    assert_allclose(value, _jac_f(x), rtol=1e-6)


# Hessian: the value rides the forward pass for free in every mode


@pytest.mark.hessian
def test_hessian_value_rides_forward_pass_for_free(hessian_mode):
    """Every Hessian mode invokes ``f`` exactly once, value-free or not.

    ``fwd_over_rev`` and ``rev_over_rev`` carry the value on the outer forward
    pass.
    ``rev_over_fwd`` lifts it out of the vmapped HVPs as the ``jax.grad`` aux
    (each inner ``jax.jvp`` already evaluates ``f``), so it no longer pays a
    dedicated ``f`` call for the value either.
    """
    x = jnp.arange(1.0, 7.0)

    f_free, c_free = _counting(_hess_f)
    fn_free = hessian(f_free, x, mode=hessian_mode)
    c_free["n"] = 0
    fn_free(x)

    f_val, c_val = _counting(_hess_f)
    fn_val = value_and_hessian(f_val, x, mode=hessian_mode)
    c_val["n"] = 0
    value, _ = fn_val(x)

    assert c_free["n"] == 1
    assert c_val["n"] == 1
    assert_allclose(value, _hess_f(x), rtol=1e-6)
