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

jac_fn = jax.jit(asdex.jacobian(f, input_shape=x.shape))
J = jac_fn(x)
```

Instead of 999 VJPs or 1000 JVPs,
`asdex` computes the full sparse Jacobian with just 2 JVPs.

## Next Steps

- [Getting Started](tutorials/getting-started.md) — step-by-step tutorial
- [How-To Guides](how-to/jacobians.md) — task-oriented recipes
- [Explanation](explanation/asd.md) — how and why it works
- [API Reference](reference/jacobian.md) — full API documentation

## Acknowledgements

This package is built with Claude Code based on previous work by [Adrian Hill](https://github.com/adrhill), [Guillaume Dalle](https://github.com/gdalle), and [Alexis Montoison](https://github.com/amontoison) in the [Julia programming language](https://julialang.org):

- [_An Illustrated Guide to Automatic Sparse Differentiation_](https://iclr-blogposts.github.io/2025/blog/sparse-autodiff/), A. Hill, G. Dalle, A. Montoison (2025)
- [_Sparser, Better, Faster, Stronger: Efficient Automatic Differentiation for Sparse Jacobians and Hessians_](https://openreview.net/forum?id=GtXSN52nIW), A. Hill & G. Dalle (2025)
- [_Revisiting Sparse Matrix Coloring and Bicoloring_](https://arxiv.org/abs/2505.07308), A. Montoison, G. Dalle, A. Gebremedhin (2025)
- [_SparseConnectivityTracer.jl_](https://github.com/adrhill/SparseConnectivityTracer.jl), A. Hill, G. Dalle
- [_SparseMatrixColorings.jl_](https://github.com/gdalle/SparseMatrixColorings.jl), G. Dalle, A. Montoison
- [_sparsediffax_](https://github.com/gdalle/sparsediffax), G. Dalle

which in turn stands on the shoulders of giants — notably Andreas Griewank, Andrea Walther, and Assefaw Gebremedhin.
