"""Graph coloring for sparse Jacobian and Hessian computation.

Greedy coloring assigns colors to vertices such that conflicting vertices
get different colors.
Row coloring enables computing multiple Jacobian rows in a single VJP.
Column coloring enables computing multiple Jacobian columns in a single JVP.
Symmetric coloring exploits Hessian symmetry for fewer colors.

Algorithms adapted from SparseMatrixColorings.jl (MIT license)
Copyright (c) 2024 Guillaume Dalle, Alexis Montoison, and contributors
https://github.com/gdalle/SparseMatrixColorings.jl
See also: Dalle & Montoison (2025), https://arxiv.org/abs/2505.07308
"""

from asdex._errors import DenseColoringWarning, InvalidColoringError
from asdex._pattern import StarSet
from asdex.coloring._api import (
    hessian_coloring,
    hessian_coloring_from_sparsity,
    jacobian_coloring,
    jacobian_coloring_from_sparsity,
)
from asdex.coloring._color_greedy import color_cols, color_rows
from asdex.coloring._color_symmetric import color_symmetric
from asdex.coloring._graph import reconstruct_edge_arrays

__all__ = [
    "DenseColoringWarning",
    "InvalidColoringError",
    "StarSet",
    "color_cols",
    "color_rows",
    "color_symmetric",
    "hessian_coloring",
    "hessian_coloring_from_sparsity",
    "jacobian_coloring",
    "jacobian_coloring_from_sparsity",
    "reconstruct_edge_arrays",
]
