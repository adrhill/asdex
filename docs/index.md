# asdex

[![CI](https://github.com/adrhill/asdex/actions/workflows/ci.yml/badge.svg)](https://github.com/adrhill/asdex/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/adrhill/asdex/graph/badge.svg)](https://codecov.io/gh/adrhill/asdex)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![PyPI](https://img.shields.io/pypi/v/asdex)](https://pypi.org/project/asdex/)

[![Contributing](https://img.shields.io/badge/guide-contributing-blueviolet)](contributing.md)
[![AI Policy](https://img.shields.io/badge/policy-AI%20usage-blueviolet)](contributing.md#ai-policy)

[![Benchmarks](https://img.shields.io/badge/benchmarks-view-blue)](https://adrianhill.de/asdex/dev/bench/)
[![Changelog](https://img.shields.io/badge/news-changelog-yellow)](https://github.com/adrhill/asdex/blob/main/CHANGELOG.md)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18788242.svg)](https://doi.org/10.5281/zenodo.18788242)

**Automatic Sparse Differentiation in JAX.**

`asdex` exploits sparsity structure to efficiently compute sparse Jacobians and Hessians.
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

```python exec="true" session="index" source="above"
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

Since sparsity detection and coloring can be expensive on large problems,
we recommend saving and reusing colored patterns:

```python exec="true" session="index" source="above"
from asdex import ColoredPattern, jacobian_coloring, jacobian_from_coloring

# Compute coloring once...
coloring = jacobian_coloring(f, x)
coloring.save("colored.npz")

# ...load and reuse later
coloring = ColoredPattern.load("colored.npz")
jac_fn = jax.jit(jacobian_from_coloring(f, coloring))
```

## Features

**The full ASD pipeline:**

- **Sparse [Jacobians](https://adrianhill.de/asdex/how-to/jacobians/) and [Hessians](https://adrianhill.de/asdex/how-to/hessians/)**: one VJP/JVP/HVP per color, with automatic (or user-defined) mode selection.
- **[Sparsity detection](https://adrianhill.de/asdex/explanation/sparsity-detection/)**: finds [global sparsity patterns](https://adrianhill.de/asdex/explanation/global-sparsity/) valid for all inputs.
- **[Graph coloring](https://adrianhill.de/asdex/explanation/coloring/)**: row, column, and symmetric coloring minimize AD passes.
- **[Correctness verification](https://adrianhill.de/asdex/how-to/verification/)** against vanilla JAX.

**You already know your sparsity pattern?**

- **[Manually provide sparsity patterns](https://adrianhill.de/asdex/how-to/jacobians/#manually-providing-a-sparsity-pattern)**: supply a known pattern from dense, COO, or BCOO formats.
- **[Precompute, save & load](https://adrianhill.de/asdex/how-to/jacobians/#precomputing-the-colored-pattern)**: reuse a colored pattern across inputs, or persist it by saving and loading.

**An interface mirroring JAX:**

- **[Multiple inputs and outputs](https://adrianhill.de/asdex/how-to/jacobians/#multiple-inputs-and-outputs)**: supports multi-argument functions via `argnums`, as well as multiple return values.
- **[PyTree inputs and outputs](https://adrianhill.de/asdex/how-to/jacobians/#pytree-inputs-and-outputs)**: sparse differentiation through arbitrary nested [PyTrees](https://docs.jax.dev/en/latest/pytrees.html).
- **[Auxiliary outputs](https://adrianhill.de/asdex/how-to/jacobians/#auxiliary-outputs)**: supports `has_aux=True` for functions returning `(output, aux)`.
- **[Value and derivative](https://adrianhill.de/asdex/how-to/jacobians/#getting-the-primal-value-too)**: `value_and_jacobian` / `value_and_hessian` return the primal value `f(x)` without a redundant forward pass.

**And more:**

- **[Multiple output formats](https://adrianhill.de/asdex/how-to/jacobians/#output-formats)**: decompression to BCOO, dense JAX arrays, NumPy, and SciPy (COO/CSR/CSC) arrays.
- **[Bounded memory](https://adrianhill.de/asdex/how-to/jacobians/#reducing-peak-memory-with-chunking)**: `chunk_size` caps parallel AD passes for large color counts.
- **[Visualizations](https://adrianhill.de/asdex/how-to/visualization/)**: `spy` plots and braille pattern previews.

## Next Steps

- [Getting Started](tutorials/getting-started.md) — step-by-step tutorial
- [How-To Guides](how-to/jacobians.md) — task-oriented recipes
- [Explanation](explanation/asd.md) — how and why it works
- [API Reference](reference/jacobian.md) — full API documentation
- [Contributing](contributing.md) — guidelines for collaborating on asdex
- [AI Policy](contributing.md#ai-policy) — guidelines for LLM contributions

## Related work

Prior work on ASD by asdex's authors Adrian Hill ([`@adrhill`](https://github.com/adrhill)) and Guillaume Dalle ([`@gdalle`](https://github.com/gdalle)),
as well as Alexis Montoison ([`@amontoison`](https://github.com/amontoison)):

- [_An Illustrated Guide to Automatic Sparse Differentiation_](https://iclr-blogposts.github.io/2025/blog/sparse-autodiff/), Hill, Dalle, Montoison (2025)
- [_Sparser, Better, Faster, Stronger: Efficient Automatic Differentiation for Sparse Jacobians and Hessians_](https://openreview.net/forum?id=GtXSN52nIW), Hill & Dalle (2025)
- [_Revisiting Sparse Matrix Coloring and Bicoloring_](https://arxiv.org/abs/2505.07308), Montoison, Dalle, Gebremedhin (2025)
- [`SparseConnectivityTracer.jl`](https://github.com/adrhill/SparseConnectivityTracer.jl), Hill & Dalle
- [`SparseMatrixColorings.jl`](https://github.com/gdalle/SparseMatrixColorings.jl), Dalle & Montoison
- [`DifferentiationInterface.jl`](https://github.com/JuliaDiff/DifferentiationInterface.jl), Dalle & Hill

Prior and concurrent (partial) attempts at ASD in JAX:

- [`sparsejac`](https://github.com/mfschubert/sparsejac): coloring and decompression
- [`sparsediffax`](https://github.com/gdalle/sparsediffax): coloring and decompression (by asdex's [`@gdalle`](https://github.com/gdalle))
- [`jax-nansparse`](https://github.com/nardi/jax-nansparse): sparsity detection using NaN propagation
- [`JAX-AMG`](https://github.com/jx-wang-s-group/JAX-AMG): specialized ASD module for algebraic multigrid methods
- [`tatva`](https://github.com/smec-ethz/tatva): specialized ASD module for FEM
- See discussion in [JAX issue #1032](https://github.com/jax-ml/jax/issues/1032)

## Acknowledgements

Adrian Hill gratefully acknowledges funding from the German Federal Ministry of Education and Research under the grant BIFOLD26B.

This package is [built with Claude Code](contributing.md#ai-policy),
based on previous, hand-written work by the same authors in the [Julia programming language](https://julialang.org), as noted above.
These works in turn stand on the shoulders of giants, notably Andreas Griewank, Andrea Walther, and Assefaw Gebremedhin.

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
