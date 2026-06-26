# asdex

[![CI](https://github.com/adrhill/asdex/actions/workflows/ci.yml/badge.svg)](https://github.com/adrhill/asdex/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/adrhill/asdex/graph/badge.svg)](https://codecov.io/gh/adrhill/asdex)
[![PyPI](https://img.shields.io/pypi/v/asdex)](https://pypi.org/project/asdex/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18788242.svg)](https://doi.org/10.5281/zenodo.18788242)
[![Benchmarks](https://img.shields.io/badge/benchmarks-view-blue)](https://adrianhill.de/asdex/dev/bench/)
[![Changelog](https://img.shields.io/badge/news-changelog-yellow)](https://github.com/adrhill/asdex/blob/main/CHANGELOG.md)

**Automatic Sparse Differentiation in JAX.**

`asdex` (pronounced _Aztecs_) exploits sparsity structure to efficiently compute sparse Jacobians and Hessians.
It implements a custom [Jaxpr](https://docs.jax.dev/en/latest/jaxpr.html) interpreter
that uses [abstract interpretation](explanation/sparsity-detection.md)
to detect [global sparsity patterns](explanation/global-sparsity.md) from the computation graph,
then uses [graph coloring](explanation/coloring.md) to minimize the number of AD passes needed.
Refer to our [*Illustrated Guide to Automatic Sparse Differentiation*](https://iclr-blogposts.github.io/2025/blog/sparse-autodiff/) for more information.

## Installation

```bash
pip install asdex
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add asdex
```

## Quick Example

```python
import asdex
import jax
import numpy as np

def f(x):
    return (x[1:] - x[:-1]) ** 2

x = np.random.randn(1000)

jac_fn = jax.jit(asdex.jacobian(f, x)) # shape and dtype of `x` are used for sparsity pattern detection
J = jac_fn(x)
```

Instead of 999 VJPs or 1000 JVPs,
`asdex` computes the full sparse Jacobian with just 2 JVPs.

## Features

**The full ASD pipeline:**
- **Efficient computation of Sparse [Jacobians](how-to/jacobians.md) and [Hessians](how-to/hessians.md)**: one VJP/JVP/HVP per color, with automatic (or user-defined) mode selection.
- **[Sparsity detection](explanation/sparsity-detection.md)**: a custom jaxpr interpreter finds [global sparsity patterns](explanation/global-sparsity.md) valid for all inputs.
- **[Graph coloring](explanation/coloring.md)**: row, column, and symmetric coloring to minimize AD passes.
- **[Correctness verification](how-to/verification.md)**: `check_jacobian_correctness` / `check_hessian_correctness` against vanilla JAX.

**You already know your sparsity pattern?**
- **[Manually provide sparsity patterns](how-to/jacobians.md#manually-providing-a-sparsity-pattern)**: supply a known pattern from dense, COO, or BCOO formats.
- **[Precompute, save & load](how-to/jacobians.md#precomputing-the-colored-pattern)**: reuse a `ColoredPattern` across inputs, or persist it with `.save()` / `.load()`.

**An interface mirroring JAX:**
- **[Multiple inputs and outputs](how-to/jacobians.md#multiple-inputs-and-outputs)**: multi-argument functions via `argnums` and multiple return values, mirroring `jax.jacobian`.
- **[PyTree inputs and outputs](how-to/jacobians.md#pytree-inputs-and-outputs)**: sparse differentiation through arbitrary nested [PyTrees](https://docs.jax.dev/en/latest/pytrees.html).
- **[Auxiliary outputs](how-to/jacobians.md#auxiliary-outputs)**: supports `has_aux=True` for functions returning `(output, aux)`.
- **[Value and derivative](how-to/jacobians.md#getting-the-primal-value-too)**: `value_and_jacobian` / `value_and_hessian` return the primal value `f(x)` without a redundant forward pass.

**And more:**
- **[Multiple output formats](how-to/jacobians.md#output-formats)**: decompression to BCOO (default), dense JAX arrays, NumPy, and SciPy (COO/CSR/CSC) arrays.
- **[Bounded memory](how-to/jacobians.md#reducing-peak-memory-with-chunking)**: `chunk_size` caps parallel AD passes for large color counts.
- **[Visualizations](how-to/visualization.md)**: `spy` plots and braille pattern previews.

## Next Steps

- [Getting Started](tutorials/getting-started.md) — step-by-step tutorial
- [How-To Guides](how-to/jacobians.md) — task-oriented recipes
- [Explanation](explanation/asd.md) — how and why it works
- [API Reference](reference/jacobian.md) — full API documentation

## Acknowledgements

Adrian Hill gratefully acknowledges funding from the German Federal Ministry of Education and Research under the grant BIFOLD26B.

This package is built with Claude Code based on previous work by [Adrian Hill](https://github.com/adrhill), [Guillaume Dalle](https://github.com/gdalle), and [Alexis Montoison](https://github.com/amontoison) in the [Julia programming language](https://julialang.org):

- [_An Illustrated Guide to Automatic Sparse Differentiation_](https://iclr-blogposts.github.io/2025/blog/sparse-autodiff/), Hill, Dalle, Montoison (2025)
- [_Sparser, Better, Faster, Stronger: Efficient Automatic Differentiation for Sparse Jacobians and Hessians_](https://openreview.net/forum?id=GtXSN52nIW), Hill & Dalle (2025)
- [_Revisiting Sparse Matrix Coloring and Bicoloring_](https://arxiv.org/abs/2505.07308), Montoison, Dalle, Gebremedhin (2025)
- [_SparseConnectivityTracer.jl_](https://github.com/adrhill/SparseConnectivityTracer.jl), Hill, Dalle
- [_SparseMatrixColorings.jl_](https://github.com/gdalle/SparseMatrixColorings.jl), Dalle, Montoison
- [_sparsediffax_](https://github.com/gdalle/sparsediffax), Dalle

which in turn stands on the shoulders of giants — notably Andreas Griewank, Andrea Walther, and Assefaw Gebremedhin.

## Citation

If you use asdex in your research, please cite:

```bibtex
@software{asdex2026,
  author = {Hill, Adrian},
  title = {asdex: Automatic Sparse Differentiation in JAX},
  url = {https://github.com/adrhill/asdex},
  doi = {10.5281/zenodo.18788242}
}
```
