"""Sparse and compressed Jacobian and Hessian computation.

The decompression stage of ``detect -> color -> decompress``,
which is itself compress (one VJP/JVP/HVP per color, producing the dense
compressed matrix ``B``) then decompress (scatter ``B`` back into the pattern).

This package exposes the public surface:
the one-shot ``jacobian``/``hessian``/``value_and_*`` family and their
``*_from_coloring`` variants, the ``compressed_*`` / ``value_and_compressed_*``
factories that stop at ``B``, and ``decompress``/``decompress_data`` that turn
``B`` back into a sparse matrix.
"""

from asdex.decompression._api import (
    compressed_hessian,
    compressed_hessian_from_coloring,
    compressed_hessian_stack_from_coloring,
    compressed_jacobian,
    compressed_jacobian_from_coloring,
    decompress,
    decompress_data,
    hessian,
    hessian_from_coloring,
    jacobian,
    jacobian_from_coloring,
    value_and_compressed_hessian,
    value_and_compressed_hessian_from_coloring,
    value_and_compressed_jacobian,
    value_and_compressed_jacobian_from_coloring,
    value_and_hessian,
    value_and_hessian_from_coloring,
    value_and_jacobian,
    value_and_jacobian_from_coloring,
)

__all__ = [
    "compressed_hessian",
    "compressed_hessian_from_coloring",
    "compressed_hessian_stack_from_coloring",
    "compressed_jacobian",
    "compressed_jacobian_from_coloring",
    "decompress",
    "decompress_data",
    "hessian",
    "hessian_from_coloring",
    "jacobian",
    "jacobian_from_coloring",
    "value_and_compressed_hessian",
    "value_and_compressed_hessian_from_coloring",
    "value_and_compressed_jacobian",
    "value_and_compressed_jacobian_from_coloring",
    "value_and_hessian",
    "value_and_hessian_from_coloring",
    "value_and_jacobian",
    "value_and_jacobian_from_coloring",
]
