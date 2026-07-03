# Full API

## Differentiation

::: asdex.jacobian
::: asdex.jacobian_from_coloring
::: asdex.value_and_jacobian
::: asdex.value_and_jacobian_from_coloring

---

::: asdex.hessian
::: asdex.hessian_from_coloring
::: asdex.value_and_hessian
::: asdex.value_and_hessian_from_coloring

## Coloring

::: asdex.jacobian_coloring
::: asdex.jacobian_coloring_from_sparsity

---

::: asdex.hessian_coloring
::: asdex.hessian_coloring_from_sparsity

---

::: asdex.color_rows
::: asdex.color_cols
::: asdex.color_symmetric

## Visualization

::: asdex.spy

## Sparsity Detection

::: asdex.jacobian_sparsity
::: asdex.hessian_sparsity

## Data Structures

::: asdex.SparsityPattern
::: asdex.ColoredPattern

---

::: asdex.JacobianMode
::: asdex.HessianMode
::: asdex.ColoringMode
::: asdex.VerificationError

## Compressed Differentiation

Advanced entry points that stop at the compressed matrix \(B\),
leaving decompression to the caller.
See [Skipping Decompression](../how-to/jacobians.md#skipping-decompression).

::: asdex.compressed_jacobian
::: asdex.compressed_jacobian_from_coloring
::: asdex.value_and_compressed_jacobian
::: asdex.value_and_compressed_jacobian_from_coloring

---

::: asdex.compressed_hessian
::: asdex.compressed_hessian_from_coloring
::: asdex.value_and_compressed_hessian
::: asdex.value_and_compressed_hessian_from_coloring

---

::: asdex.decompress
::: asdex.decompress_data
