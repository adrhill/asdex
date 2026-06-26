<p align="center">
  <img src="docs/assets/logo.svg" alt="asdex logo" width="200">
</p>
<h1 align="center">asdex</h1>
<p align="center">
  <a href="https://iclr-blogposts.github.io/2025/blog/sparse-autodiff/">Automatic Sparse Differentiation</a> in JAX.
</p>

<p align="center">
  <a href="https://github.com/adrhill/asdex/actions/workflows/ci.yml"><img src="https://github.com/adrhill/asdex/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/adrhill/asdex"><img src="https://codecov.io/gh/adrhill/asdex/graph/badge.svg" alt="codecov"></a>
  <a href="https://pypi.org/project/asdex/"><img src="https://img.shields.io/pypi/v/asdex" alt="PyPI"></a>
  <a href="https://doi.org/10.5281/zenodo.18788242"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.18788242.svg" alt="DOI"></a>
</p>
<p align="center">
  <a href="https://adrianhill.de/asdex/"><img src="https://img.shields.io/badge/docs-online-blue" alt="Docs"></a>
  <a href="https://adrianhill.de/asdex/dev/bench/"><img src="https://img.shields.io/badge/benchmarks-view-blue" alt="Benchmarks"></a>
  <a href="https://github.com/adrhill/asdex/blob/main/CHANGELOG.md"><img src="https://img.shields.io/badge/news-changelog-yellow" alt="Changelog"></a>
</p>


`asdex` (pronounced _Aztecs_) exploits sparsity structure to efficiently materialize Jacobians and Hessians.
It implements a custom [Jaxpr](https://docs.jax.dev/en/latest/jaxpr.html) interpreter
that uses [abstract interpretation](https://en.wikipedia.org/wiki/Abstract_interpretation)
to detect sparsity patterns from the computation graph,
then uses graph coloring to minimize the number of AD passes needed.

## Installation

```bash
pip install asdex
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add asdex
```

## Example

```python
import asdex
import jax
import jax.numpy as jnp

def f(x):
    return (x[1:] - x[:-1]) ** 2

x_sample = jnp.zeros(50)  # sample input for sparsity pattern detection
jac_fn = jax.jit(asdex.jacobian(f, x_sample))
# ColoredPattern(49×50, nnz=98, sparsity=96.0%, JVP, 2 colors)
#   2 JVPs (instead of 49 VJPs or 50 JVPs)
# ⎡⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎤   ⎡⣿⎤
# ⎢⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥   ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥   ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥   ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥   ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥   ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥ → ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⎥   ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⎥   ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⎥   ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⎥   ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⎥   ⎢⣿⎥
# ⎣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⎦   ⎣⠉⎦

for x in inputs:
    J = jac_fn(x)
```

Instead of 49 VJPs or 50 JVPs,
`asdex` computes the full sparse Jacobian with just 2 JVPs.

## Features

**The full ASD pipeline:**
- **Efficient computation of Sparse [Jacobians](https://adrianhill.de/asdex/how-to/jacobians/) and [Hessians](https://adrianhill.de/asdex/how-to/hessians/)**: one VJP/JVP/HVP per color, with automatic (or user-defined) mode selection.
- **[Sparsity detection](https://adrianhill.de/asdex/explanation/sparsity-detection/)**: a custom jaxpr interpreter finds [global sparsity patterns](https://adrianhill.de/asdex/explanation/global-sparsity/) valid for all inputs.
- **[Graph coloring](https://adrianhill.de/asdex/explanation/coloring/)**: row, column, and symmetric coloring to minimize AD passes.
- **[Correctness verification](https://adrianhill.de/asdex/how-to/verification/)**: `check_jacobian_correctness` / `check_hessian_correctness` against vanilla JAX.

**You already know your sparsity pattern?**
- **[Manually provide sparsity patterns](https://adrianhill.de/asdex/how-to/jacobians/#manually-providing-a-sparsity-pattern)**: supply a known pattern from dense, COO, or BCOO formats.
- **[Precompute, save & load](https://adrianhill.de/asdex/how-to/jacobians/#precomputing-the-colored-pattern)**: reuse a `ColoredPattern` across inputs, or persist it with `.save()` / `.load()`.

**An interface mirroring JAX:**
- **[Multiple inputs and outputs](https://adrianhill.de/asdex/how-to/jacobians/#multiple-inputs-and-outputs)**: multi-argument functions via `argnums` and multiple return values, mirroring `jax.jacobian`.
- **[PyTree inputs and outputs](https://adrianhill.de/asdex/how-to/jacobians/#pytree-inputs-and-outputs)**: sparse differentiation through arbitrary nested [PyTrees](https://docs.jax.dev/en/latest/pytrees.html).
- **[Auxiliary outputs](https://adrianhill.de/asdex/how-to/jacobians/#auxiliary-outputs)**: supports `has_aux=True` for functions returning `(output, aux)`.
- **[Value and derivative](https://adrianhill.de/asdex/how-to/jacobians/#getting-the-primal-value-too)**: `value_and_jacobian` / `value_and_hessian` return the primal value `f(x)` without a redundant forward pass.

**And more:**
- **[Multiple output formats](https://adrianhill.de/asdex/how-to/jacobians/#output-formats)**: decompression to BCOO (default), dense JAX arrays, NumPy, and SciPy (COO/CSR/CSC) arrays.
- **[Bounded memory](https://adrianhill.de/asdex/how-to/jacobians/#reducing-peak-memory-with-chunking)**: `chunk_size` caps parallel AD passes for large color counts.
- **[Visualizations](https://adrianhill.de/asdex/how-to/visualization/)**: `spy` plots and braille pattern previews.

## Documentation

- [Getting Started](https://adrianhill.de/asdex/tutorials/getting-started/): step-by-step tutorial
- [How-To Guides](https://adrianhill.de/asdex/how-to/jacobians/): task-oriented recipes
- [Explanation](https://adrianhill.de/asdex/explanation/sparsity-detection/): how and why it works
- [API Reference](https://adrianhill.de/asdex/reference/): full API documentation

## Acknowledgements

Adrian Hill gratefully acknowledges funding from the German Federal Ministry of Education and Research under the grant BIFOLD26B.

This package is built with Claude Code based on previous work by [Adrian Hill](https://github.com/adrhill), [Guillaume Dalle](https://github.com/gdalle), and [Alexis Montoison](https://github.com/amontoison) in the [Julia programming language](https://julialang.org):

- [_An Illustrated Guide to Automatic Sparse Differentiation_](https://iclr-blogposts.github.io/2025/blog/sparse-autodiff/), Hill, Dalle, Montoison (2025)
- [_Sparser, Better, Faster, Stronger: Efficient Automatic Differentiation for Sparse Jacobians and Hessians_](https://openreview.net/forum?id=GtXSN52nIW), Hill & Dalle (2025)
- [_Revisiting Sparse Matrix Coloring and Bicoloring_](https://arxiv.org/abs/2505.07308), Montoison, Dalle, Gebremedhin (2025)
- [_SparseConnectivityTracer.jl_](https://github.com/adrhill/SparseConnectivityTracer.jl), Hill, Dalle
- [_SparseMatrixColorings.jl_](https://github.com/gdalle/SparseMatrixColorings.jl), Dalle, Montoison
- [_sparsediffax_](https://github.com/gdalle/sparsediffax), Dalle

These works in turn stand on the shoulders of giants, notably Andreas Griewank, Andrea Walther, and Assefaw Gebremedhin.

The asdex logo was designed by [@overripemango](https://instagram.com/overripemango).

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
