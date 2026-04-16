"""Jacobian and Hessian sparsity detection via jaxpr graph analysis."""

from asdex.detection._api import (
    hessian_sparsity,
    jacobian_sparsity,
)

__all__ = [
    "hessian_sparsity",
    "jacobian_sparsity",
]
