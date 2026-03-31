"""Test Jacobian sparsity by comparing against SymPy symbolic derivatives.

This module generates random mathematical expressions using SymPy primitives,
converts them to JAX functions, and verifies that asdex's sparsity detection
matches the ground truth computed from symbolic differentiation.
"""

import random
from collections.abc import Callable
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp
from sympy import (
    Abs,
    Symbol,
    asinh,
    atan,
    cos,
    cosh,
    exp,
    log,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
)

from asdex import hessian_sparsity, jacobian_sparsity

# Type alias for JAX functions
JaxFn = Callable[[jnp.ndarray], jnp.ndarray]

# Unary functions that preserve non-zero derivatives
UNARY_OPS = [sin, cos, tan, exp, sqrt, sinh, cosh, tanh, Abs, log, atan, asinh]

# Unary ops for Hessian tests (excludes Abs due to SymPy's complex-analytic treatment)
UNARY_OPS_HESSIAN = [sin, cos, tan, exp, sqrt, sinh, cosh, tanh, log, atan, asinh]

# Binary operations
BINARY_OPS = ["add", "sub", "mul", "div"]


class SympyToJax:
    """Convert SymPy expressions to JAX functions using dispatch on func type."""

    # Mapping from SymPy function types to JAX functions
    UNARY_MAP: ClassVar[dict[type, Callable]] = {
        sp.sin: jnp.sin,
        sp.cos: jnp.cos,
        sp.tan: jnp.tan,
        sp.exp: jnp.exp,
        sp.log: jnp.log,
        sp.Abs: jnp.abs,
        sp.sinh: jnp.sinh,
        sp.cosh: jnp.cosh,
        sp.tanh: jnp.tanh,
        sp.asin: jnp.arcsin,
        sp.acos: jnp.arccos,
        sp.atan: jnp.arctan,
        sp.asinh: jnp.arcsinh,
    }

    def __init__(self, symbols: list[Symbol]):
        """Initialize converter with symbol-to-index mapping."""
        self.sym_to_idx = {s: i for i, s in enumerate(symbols)}

    def convert(self, e: sp.Basic) -> JaxFn:
        """Convert a SymPy expression to a JAX function."""
        # Handle symbols
        if isinstance(e, sp.Symbol):
            idx = self.sym_to_idx[e]
            return lambda x, i=idx: x[i]

        # Handle numbers
        if e.is_number:
            val = float(e.evalf())  # ty: ignore[unresolved-attribute]
            return lambda x, v=val: v

        # Handle unary functions via lookup
        if e.func in self.UNARY_MAP:
            jax_fn = self.UNARY_MAP[e.func]
            arg_fn = self.convert(e.args[0])
            return lambda x, f=arg_fn, g=jax_fn: g(f(x))

        # Handle Add
        if e.func == sp.Add:
            term_fns = [self.convert(arg) for arg in e.args]
            return lambda x, fns=term_fns: sum(f(x) for f in fns)

        # Handle Mul
        if e.func == sp.Mul:
            factor_fns = [self.convert(arg) for arg in e.args]

            def mul_fn(x, fns=factor_fns):
                result = fns[0](x)
                for f in fns[1:]:
                    result = result * f(x)
                return result

            return mul_fn

        # Handle Pow (includes sqrt as Pow(x, 1/2))
        if e.func == sp.Pow:
            base_fn = self.convert(e.args[0])
            if e.args[1].is_number:
                exp_val = float(e.args[1].evalf())  # ty: ignore[unresolved-attribute]
                return lambda x, b=base_fn, p=exp_val: b(x) ** p
            exp_fn = self.convert(e.args[1])
            return lambda x, b=base_fn, p=exp_fn: b(x) ** p(x)

        raise NotImplementedError(f"Unsupported: {type(e)} with func={e.func} - {e}")


def sympy_to_jax_fn(
    exprs: list[sp.Expr], symbols: list[Symbol]
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Convert a list of SymPy expressions to a JAX function.

    Args:
        exprs: List of SymPy expressions (output components)
        symbols: List of SymPy symbols (input variables)

    Returns:
        JAX function f(x) -> array
    """
    converter = SympyToJax(symbols)
    expr_fns = [converter.convert(expr) for expr in exprs]

    def jax_fn(x: jnp.ndarray) -> jnp.ndarray:
        results = [fn(x) for fn in expr_fns]
        return jnp.array(results)

    return jax_fn


def compute_symbolic_sparsity(
    exprs: list[sp.Expr], symbols: list[Symbol]
) -> np.ndarray:
    """Compute the Jacobian sparsity pattern using SymPy symbolic differentiation.

    Args:
        exprs: List of SymPy expressions (output components)
        symbols: List of SymPy symbols (input variables)

    Returns:
        Boolean numpy array of shape (len(exprs), len(symbols))
        where entry (i,j) is True if expr[i] depends on symbols[j]
    """
    m, n = len(exprs), len(symbols)
    sparsity = np.zeros((m, n), dtype=bool)

    for i, expr in enumerate(exprs):
        for j, sym in enumerate(symbols):
            # Compute symbolic derivative
            deriv = sp.diff(expr, sym)
            # Check if derivative is structurally non-zero
            # Use simplify() which is faster than equals(0) for complex expressions
            simplified = sp.simplify(deriv)
            sparsity[i, j] = simplified != 0

    return sparsity


def random_unary_expr(
    base_expr: sp.Expr, rng: random.Random, ops: list | None = None
) -> sp.Expr:
    """Apply a random unary operation to an expression."""
    if ops is None:
        ops = UNARY_OPS
    op = rng.choice(ops)
    # Some functions need domain restrictions
    if op in (sqrt, log):
        return op(Abs(base_expr) + 1)
    return op(base_expr)


def random_binary_expr(expr1: sp.Expr, expr2: sp.Expr, rng: random.Random) -> sp.Expr:
    """Combine two expressions with a random binary operation."""
    op = rng.choice(BINARY_OPS)
    if op == "add":
        return expr1 + expr2
    if op == "sub":
        return expr1 - expr2
    if op == "mul":
        return expr1 * expr2
    if op == "div":
        # Avoid division by zero
        return expr1 / (Abs(expr2) + 1)
    raise ValueError(f"Unknown operation: {op}")


def generate_random_expr(
    symbols: list[Symbol],
    depth: int,
    rng: random.Random,
    unary_ops: list | None = None,
) -> sp.Expr:
    """Generate a random SymPy expression.

    Args:
        symbols: Available input symbols
        depth: Maximum recursion depth
        rng: Random number generator
        unary_ops: List of unary operations to use (defaults to UNARY_OPS)

    Returns:
        A random SymPy expression
    """
    if depth == 0 or rng.random() < 0.3:
        # Base case: return a symbol or constant
        if rng.random() < 0.8:
            return rng.choice(symbols)
        return sp.Integer(rng.randint(1, 5))

    # Recursive case: build more complex expression
    choice = rng.random()

    if choice < 0.4:
        # Unary operation
        sub_expr = generate_random_expr(symbols, depth - 1, rng, unary_ops)
        return random_unary_expr(sub_expr, rng, unary_ops)
    # Binary operation
    left = generate_random_expr(symbols, depth - 1, rng, unary_ops)
    right = generate_random_expr(symbols, depth - 1, rng, unary_ops)
    return random_binary_expr(left, right, rng)


def generate_random_function(
    n_inputs: int,
    n_outputs: int,
    max_depth: int,
    seed: int,
) -> tuple[list[sp.Expr], list[Symbol]]:
    """Generate a random vector-valued function.

    Args:
        n_inputs: Number of input variables
        n_outputs: Number of output components
        max_depth: Maximum expression tree depth
        seed: Random seed for reproducibility

    Returns:
        Tuple of (output expressions, input symbols)
    """
    rng = random.Random(seed)
    symbols = [Symbol(f"x{i}") for i in range(n_inputs)]
    exprs = [generate_random_expr(symbols, max_depth, rng) for _ in range(n_outputs)]
    return exprs, symbols


class TestSympyComparison:
    """Test suite comparing asdex against SymPy symbolic derivatives."""

    def _run_comparison(
        self, n_inputs: int, n_outputs: int, max_depth: int, seed: int
    ) -> None:
        """Run a single comparison test.

        Args:
            n_inputs: Number of input variables
            n_outputs: Number of output components
            max_depth: Maximum expression depth
            seed: Random seed
        """
        exprs, symbols = generate_random_function(n_inputs, n_outputs, max_depth, seed)

        # Compute ground truth sparsity from SymPy
        expected = compute_symbolic_sparsity(exprs, symbols)

        # Convert to JAX and compute sparsity with asdex
        jax_fn = sympy_to_jax_fn(exprs, symbols)
        result = jacobian_sparsity(jax_fn, n_inputs).todense().astype(bool)

        # asdex sparsity should exactly match symbolic sparsity
        if not np.array_equal(expected, result):
            expr_strs = [str(e) for e in exprs]
            missed = expected & ~result
            extra = result & ~expected
            raise AssertionError(
                f"Sparsity mismatch!\n"
                f"Expressions: {expr_strs}\n"
                f"Expected:\n{expected.astype(int)}\n"
                f"Got:\n{result.astype(int)}\n"
                f"Missed (false negatives):\n{missed.astype(int)}\n"
                f"Extra (false positives):\n{extra.astype(int)}"
            )

    def test_simple_expressions(self):
        """Test simple expressions with depth 1."""
        for seed in range(10):
            self._run_comparison(n_inputs=3, n_outputs=2, max_depth=1, seed=seed)

    def test_medium_expressions(self):
        """Test medium complexity expressions with depth 2."""
        for seed in range(10):
            self._run_comparison(n_inputs=4, n_outputs=3, max_depth=2, seed=seed)

    def test_complex_expressions(self):
        """Test complex expressions with depth 3."""
        for seed in range(5):
            self._run_comparison(n_inputs=4, n_outputs=3, max_depth=3, seed=seed)

    def test_wide_inputs(self):
        """Test with many input variables."""
        for seed in range(5):
            self._run_comparison(n_inputs=8, n_outputs=2, max_depth=2, seed=seed)

    @pytest.mark.slow
    def test_wide_outputs(self):
        """Test with many output components."""
        for seed in range(5):
            self._run_comparison(n_inputs=3, n_outputs=6, max_depth=2, seed=seed)


# Specific regression tests for edge cases
class TestSympyEdgeCases:
    """Test specific edge cases using SymPy verification."""

    def test_nested_unary(self):
        """Test deeply nested unary operations."""
        x = Symbol("x")
        expr = sin(cos(exp(tanh(x))))
        expected = compute_symbolic_sparsity([expr], [x])

        def f(arr):
            return jnp.array([jnp.sin(jnp.cos(jnp.exp(jnp.tanh(arr[0]))))])

        result = jacobian_sparsity(f, 1).todense().astype(bool)
        np.testing.assert_array_equal(result, expected)

    def test_shared_subexpression(self):
        """Test expressions that share subexpressions."""
        x, y = Symbol("x"), Symbol("y")
        shared = x * y
        expr1 = sin(shared)
        expr2 = cos(shared) + x
        expected = compute_symbolic_sparsity([expr1, expr2], [x, y])

        def f(arr):
            shared = arr[0] * arr[1]
            return jnp.array([jnp.sin(shared), jnp.cos(shared) + arr[0]])

        result = jacobian_sparsity(f, 2).todense().astype(bool)
        np.testing.assert_array_equal(result, expected)

    def test_polynomial(self):
        """Test polynomial expressions."""
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        expr = x**2 + 2 * x * y + y**2 + z
        expected = compute_symbolic_sparsity([expr], [x, y, z])

        def f(arr):
            return jnp.array([arr[0] ** 2 + 2 * arr[0] * arr[1] + arr[1] ** 2 + arr[2]])

        result = jacobian_sparsity(f, 3).todense().astype(bool)
        np.testing.assert_array_equal(result, expected)

    def test_rational_function(self):
        """Test rational function (division)."""
        x, y = Symbol("x"), Symbol("y")
        expr = (x**2 + y) / (Abs(x) + 1)
        expected = compute_symbolic_sparsity([expr], [x, y])

        def f(arr):
            return jnp.array([(arr[0] ** 2 + arr[1]) / (jnp.abs(arr[0]) + 1)])

        result = jacobian_sparsity(f, 2).todense().astype(bool)
        np.testing.assert_array_equal(result, expected)

    def test_mixed_binary_ops(self):
        """Test mix of all binary operations."""
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        expr = (x + y) * (y - z) / (Abs(x * z) + 1)
        expected = compute_symbolic_sparsity([expr], [x, y, z])

        def f(arr):
            return jnp.array(
                [(arr[0] + arr[1]) * (arr[1] - arr[2]) / (jnp.abs(arr[0] * arr[2]) + 1)]
            )

        result = jacobian_sparsity(f, 3).todense().astype(bool)
        np.testing.assert_array_equal(result, expected)


# Hessian sparsity tests


def sympy_to_jax_scalar_fn(
    expr: sp.Expr, symbols: list[Symbol]
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Convert a scalar SymPy expression to a JAX function.

    Args:
        expr: A scalar SymPy expression
        symbols: List of SymPy symbols (input variables)

    Returns:
        JAX function f(x) -> scalar
    """
    converter = SympyToJax(symbols)
    expr_fn = converter.convert(expr)

    def jax_fn(x: jnp.ndarray) -> jnp.ndarray:
        return expr_fn(x)

    return jax_fn


def compute_symbolic_hessian_sparsity(
    expr: sp.Expr, symbols: list[Symbol]
) -> np.ndarray:
    """Compute the Hessian sparsity pattern using SymPy symbolic differentiation.

    Args:
        expr: A scalar SymPy expression
        symbols: List of SymPy symbols (input variables)

    Returns:
        Boolean numpy array of shape (n, n) where entry (i,j) is True
        if the second derivative d²f/dx_i dx_j is non-zero
    """
    n = len(symbols)
    sparsity = np.zeros((n, n), dtype=bool)

    for i, sym_i in enumerate(symbols):
        for j, sym_j in enumerate(symbols):
            # Compute second derivative
            deriv = sp.diff(sp.diff(expr, sym_i), sym_j)
            simplified = sp.simplify(deriv)
            sparsity[i, j] = simplified != 0

    return sparsity


def generate_random_scalar_function(
    n_inputs: int,
    max_depth: int,
    seed: int,
) -> tuple[sp.Expr, list[Symbol]]:
    """Generate a random scalar-valued function for Hessian testing.

    Uses UNARY_OPS_HESSIAN which excludes Abs (its symbolic second derivative
    involves complex analysis that differs from numerical computation).

    Args:
        n_inputs: Number of input variables
        max_depth: Maximum expression tree depth
        seed: Random seed for reproducibility

    Returns:
        Tuple of (scalar expression, input symbols)
    """
    rng = random.Random(seed)
    symbols = [Symbol(f"x{i}") for i in range(n_inputs)]
    expr = generate_random_expr(symbols, max_depth, rng, UNARY_OPS_HESSIAN)
    return expr, symbols


@pytest.mark.hessian
class TestHessianSympyComparison:
    """Test Hessian sparsity by comparing against SymPy symbolic derivatives.

    Due to conservative handling of the `pad` primitive in gradient jaxprs,
    asdex may over-estimate Hessian sparsity. We verify that the result is
    a superset of the true sparsity (no false negatives).
    """

    def _run_comparison(self, n_inputs: int, max_depth: int, seed: int) -> None:
        """Run a single Hessian comparison test.

        Args:
            n_inputs: Number of input variables
            max_depth: Maximum expression depth
            seed: Random seed
        """
        expr, symbols = generate_random_scalar_function(n_inputs, max_depth, seed)

        # Compute ground truth sparsity from SymPy
        expected = compute_symbolic_hessian_sparsity(expr, symbols)

        # Convert to JAX and compute sparsity with asdex
        jax_fn = sympy_to_jax_scalar_fn(expr, symbols)
        result = hessian_sparsity(jax_fn, n_inputs).todense().astype(bool)

        # asdex should have no false negatives (may have false positives due to
        # conservative pad handling)
        missed = expected & ~result
        if missed.any():
            raise AssertionError(
                f"Hessian sparsity has false negatives!\n"
                f"Expression: {expr}\n"
                f"Expected:\n{expected.astype(int)}\n"
                f"Got:\n{result.astype(int)}\n"
                f"Missed (false negatives):\n{missed.astype(int)}"
            )

    def test_simple_expressions(self):
        """Test simple expressions with depth 1."""
        for seed in range(10):
            self._run_comparison(n_inputs=3, max_depth=1, seed=seed)

    @pytest.mark.slow
    def test_medium_expressions(self):
        """Test medium complexity expressions with depth 2."""
        for seed in range(10):
            self._run_comparison(n_inputs=4, max_depth=2, seed=seed)

    @pytest.mark.slow
    def test_wide_inputs(self):
        """Test with many input variables."""
        for seed in range(5):
            self._run_comparison(n_inputs=6, max_depth=2, seed=seed)


@pytest.mark.hessian
class TestHessianSympyEdgeCases:
    """Test specific Hessian edge cases using SymPy verification."""

    def test_linear_function(self):
        """Linear functions have zero Hessian."""
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        expr = 2 * x + 3 * y - z
        expected = compute_symbolic_hessian_sparsity(expr, [x, y, z])

        def f(arr):
            return 2 * arr[0] + 3 * arr[1] - arr[2]

        result = hessian_sparsity(f, 3).todense().astype(bool)
        # Linear function has zero Hessian
        assert not expected.any()
        assert not result.any()

    def test_quadratic_diagonal(self):
        """f(x) = x² + y² + z² has diagonal Hessian."""
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        expr = x**2 + y**2 + z**2
        expected = compute_symbolic_hessian_sparsity(expr, [x, y, z])

        def f(arr):
            return arr[0] ** 2 + arr[1] ** 2 + arr[2] ** 2

        result = hessian_sparsity(f, 3).todense().astype(bool)
        # No false negatives
        missed = expected & ~result
        assert not missed.any()

    def test_cross_terms(self):
        """f(x) = x*y + y*z has off-diagonal Hessian entries."""
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        expr = x * y + y * z
        expected = compute_symbolic_hessian_sparsity(expr, [x, y, z])

        def f(arr):
            return arr[0] * arr[1] + arr[1] * arr[2]

        result = hessian_sparsity(f, 3).todense().astype(bool)
        # No false negatives
        missed = expected & ~result
        assert not missed.any()

    def test_nested_nonlinear(self):
        """Test nested nonlinear function."""
        x, y = Symbol("x"), Symbol("y")
        expr = sin(x * y) + exp(x)
        expected = compute_symbolic_hessian_sparsity(expr, [x, y])

        def f(arr):
            return jnp.sin(arr[0] * arr[1]) + jnp.exp(arr[0])

        result = hessian_sparsity(f, 2).todense().astype(bool)
        # No false negatives
        missed = expected & ~result
        assert not missed.any()

    def test_rational_hessian(self):
        """Test Hessian of rational function."""
        x, y = Symbol("x"), Symbol("y")
        expr = (x * y) / (Abs(x) + 1)
        expected = compute_symbolic_hessian_sparsity(expr, [x, y])

        def f(arr):
            return (arr[0] * arr[1]) / (jnp.abs(arr[0]) + 1)

        result = hessian_sparsity(f, 2).todense().astype(bool)
        # No false negatives
        missed = expected & ~result
        assert not missed.any()
