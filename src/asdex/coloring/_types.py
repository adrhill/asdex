"""Shared types for the coloring package.

In SparseMatrixColorings.jl, ``InvalidColoringError`` is defined in ``coloring.jl``.
``StarSet`` lives in [`asdex._pattern`][] alongside ``ColoredPattern``, which owns it.

Algorithms adapted from SparseMatrixColorings.jl (MIT license)
Copyright (c) 2024 Guillaume Dalle, Alexis Montoison, and contributors
https://github.com/gdalle/SparseMatrixColorings.jl
See also: Dalle & Montoison (2025), https://arxiv.org/abs/2505.07308

- https://github.com/gdalle/SparseMatrixColorings.jl/blob/main/src/coloring.jl
"""


class DenseColoringWarning(UserWarning):
    """Coloring uses as many colors as the dense baseline.

    Raised when sparse differentiation offers no speedup over dense differentiation.
    """


class InvalidColoringError(ValueError):
    """Raised when a user-supplied coloring violates a star-coloring constraint.

    See [`color_symmetric`][asdex.color_symmetric] with ``forced_colors``.
    """
