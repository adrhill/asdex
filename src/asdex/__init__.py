"""asdex - Global Jacobian and Hessian sparsity detection via jaxpr graph analysis.

This is global sparsity detection - asdex analyzes the computation graph
structure without evaluating derivatives, so results are valid for all inputs.
"""

from asdex._plotting import spy
from asdex.coloring import (
    DenseColoringWarning,
    color_cols,
    color_rows,
    color_symmetric,
    hessian_coloring,
    hessian_coloring_from_sparsity,
    jacobian_coloring,
    jacobian_coloring_from_sparsity,
)
from asdex.decompression import (
    hessian,
    hessian_from_coloring,
    jacobian,
    jacobian_from_coloring,
    value_and_hessian,
    value_and_hessian_from_coloring,
    value_and_jacobian,
    value_and_jacobian_from_coloring,
)
from asdex.detection import hessian_sparsity, jacobian_sparsity
from asdex.modes import HessianMode, JacobianMode, OutputFormat
from asdex.pattern import ColoredPattern, SparsityPattern
from asdex.verify import (
    VerificationError,
    check_hessian_correctness,
    check_jacobian_correctness,
)

__all__ = [
    "ColoredPattern",
    "DenseColoringWarning",
    "HessianMode",
    "JacobianMode",
    "OutputFormat",
    "SparsityPattern",
    "VerificationError",
    "check_hessian_correctness",
    "check_jacobian_correctness",
    "color_cols",
    "color_rows",
    "color_symmetric",
    "hessian",
    "hessian_coloring",
    "hessian_coloring_from_sparsity",
    "hessian_from_coloring",
    "hessian_sparsity",
    "jacobian",
    "jacobian_coloring",
    "jacobian_coloring_from_sparsity",
    "jacobian_from_coloring",
    "jacobian_sparsity",
    "spy",
    "value_and_hessian",
    "value_and_hessian_from_coloring",
    "value_and_jacobian",
    "value_and_jacobian_from_coloring",
]
