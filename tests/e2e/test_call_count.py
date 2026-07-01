"""Forward-evaluation counts for the one-shot Jacobian and Hessian APIs.

These pin how many times the user function ``f`` is invoked per call, so a
value-free API never pays for a primal value it discards.
The value rides the AD forward pass for free in every mode except the Hessian's
``rev_over_fwd``, whose forward passes happen inside the vmapped HVPs and so
cannot be lifted out.
There the value costs one dedicated ``f`` call that only ``value_and_hessian``
should pay.
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
    """Value-free and value-returning Jacobians invoke ``f`` equally often.

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

    assert c_free["n"] == c_val["n"]
    assert_allclose(value, _jac_f(x), rtol=1e-6)


# Hessian: the value is free except in rev_over_fwd


@pytest.mark.hessian
def test_hessian_value_free_skips_discarded_value_call(hessian_mode):
    """Value-free Hessian skips the primal ``f`` call unless the mode needs it.

    ``fwd_over_rev`` and ``rev_over_rev`` carry the value on the outer forward
    pass, so both variants call ``f`` equally often.
    ``rev_over_fwd`` cannot, so only ``value_and_hessian`` pays the extra call
    and the value-free path invokes ``f`` strictly fewer times.
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

    # Only rev_over_fwd cannot lift the value out of the vmapped HVPs.
    value_rides_forward_pass = hessian_mode != "rev_over_fwd"
    if value_rides_forward_pass:
        assert c_free["n"] == c_val["n"]
    else:
        assert c_free["n"] < c_val["n"]
    assert_allclose(value, _hess_f(x), rtol=1e-6)
