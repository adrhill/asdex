---
title: 'asdex: Automatic Sparse Differentiation in JAX'
tags:
  - Python
  - JAX
  - automatic differentiation
  - algorithmic differentiation
  - automatic sparse differentiation
  - sparsity pattern detection
  - graph coloring
  - sparsity
  - sparse matrices
  - Jacobians
  - Hessians
  - machine learning
  - scientific computing
authors:
  - name: Adrian Hill
    orcid: 0009-0009-5977-301X
    corresponding: true
    email: hill@tu-berlin.de
    affiliation: "1, 2"
  - name: Guillaume Dalle
    orcid: 0000-0003-4866-1687
    affiliation: 3
affiliations:
  - name: BIFOLD – Berlin Institute for the Foundations of Learning and Data, Berlin, Germany
    index: 1
  - name: Machine Learning Group, Technical University of Berlin, Berlin, Germany
    index: 2
  - name: LVMT, ENPC, Institut Polytechnique de Paris, Univ Gustave Eiffel, Marne-la-Vallée, France
    index: 3
date: 8 October 2026
bibliography: paper.bib
---

# Summary

Many tasks in scientific computing and machine learning require the Jacobian or Hessian matrix of a function.
Automatic differentiation (AD) computes these derivatives to machine precision [@griewank2008book; @baydin2018ad_survey; @blondel2026book],
but materializing a dense $m \times n$ Jacobian requires $n$ forward-mode or $m$ reverse-mode AD passes, one per column or row.
For a large class of functions, each output depends on only a few inputs, making the derivative matrix *sparse*.
Automatic sparse differentiation (ASD) exploits this structure in four steps [@hill2025illustrated]:
*detection* of the input-agnostic sparsity pattern,
*coloring* of a graph to group columns or rows that can share an AD pass,
*compressed differentiation* to compute a compressed derivative matrix with one AD pass per color,
and finally *decompression* into the original sparsity pattern.
The number of colors, and hence of AD passes, is often independent of the problem dimension:
a banded Jacobian with $b$ contiguous bands, for instance, only ever requires $b$ colors, regardless of its size.
`asdex` offers the first standalone ASD toolkit in the popular JAX [@bradbury2018jax] ecosystem.
With `asdex.jacobian` and `asdex.hessian`, it provides sparse drop-in replacements for `jax.jacobian` and `jax.hessian`.

# Statement of need

Sparse derivative matrices arise in nonlinear systems of equations, second-order optimization algorithms, sensitivity analysis, and many other applications.
The target audience for `asdex` is researchers and practitioners in scientific machine learning
who wish to leverage JAX's performance, JIT compilation, and accelerator support.

JAX's built-in `jacfwd`, `jacrev`, and `hessian` functions materialize *dense* derivative matrices, performing one AD pass per input or output dimension.
This standard approach quickly hits two main limitations:
a *memory* bottleneck, if the dense matrix is too large to store,
and a high *computational* cost, as the number of AD passes scales with the matrix dimension.
Computing a sparse Jacobian [@curtis1974sparse_jacobian] or Hessian [@powell1979hessian; @coleman1984hessian] with ASD alleviates both of these hurdles
by combining more efficient matrix storage with a faster way to fill it.
See @gebremedhin2005what_color and @griewank2008book [Chapter 8] for thorough reviews of this topic.

# State of the field

ASD initially matured in low-level programming languages such as Fortran and C++,
but we restrict our review to high-level languages,
which machine learning research favors for the rapid prototyping workflows they enable.

A general-purpose ASD implementation in a high-level, open-source language recently appeared
in Julia [@bezanson2017julia], through the combination of
`DifferentiationInterface.jl` [@dalle2025di_paper],
`SparseConnectivityTracer.jl` [@hill2025sct_paper] for sparsity pattern detection,
and `SparseMatrixColorings.jl` (SMC) [@montoison2025revisiting_coloring] for coloring.
These packages, to which the present authors contributed,
now serve as core sparse differentiation infrastructure for downstream software such as `NonlinearSolve.jl` [@pal2026nonlinearsolve].
The speedups they enable are documented in @hill2025sct_paper.

No general-purpose equivalent exists for JAX, although there are a few prototypes.
We list them in the table below with their main features:

+------------------------------------+-------------------------------------+-------------------------------------------------------------------------+
| Package                            | Detection                           | Coloring                                                                |
+====================================+=====================================+=========================================================================+
| `sparsejac`[^gh-sparsejac]         | None                                | Distance-1 (from `networkx`)                                            |
+------------------------------------+-------------------------------------+-------------------------------------------------------------------------+
| `sparsediffax`[^gh-sparsediffax]   | None                                | Distance-2 & star (from SMC)                                            |
+------------------------------------+-------------------------------------+-------------------------------------------------------------------------+
| `jax-nansparse`[^gh-jax-nansparse] | `NaN` tracing                       | None                                                                    |
+------------------------------------+-------------------------------------+-------------------------------------------------------------------------+
| `jax2sympy`[^gh-jax2sympy]         | `jaxpr` symbolic conversion         | None                                                                    |
+------------------------------------+-------------------------------------+-------------------------------------------------------------------------+
| `asdex`                            | `jaxpr` tracing                     | Distance-2 & star (based on SMC)                                        |
+------------------------------------+-------------------------------------+-------------------------------------------------------------------------+

A `jaxpr`-based approach to sparsity pattern detection has been outlined by @simpson2024sparse_jax_blog.
Parts of the ASD pipeline have also been reimplemented internally by application-specific libraries.
The sparse linear solver `JAX-AMG`[^gh-jax-amg] [@liu2026jax_amg] combines `jaxpr`-based sparsity detection with a parallel greedy distance-1 coloring,
while the finite element framework `tatva`[^gh-tatva] [@pundir2026tatva] pairs `jaxpr`-based detection with greedy distance-2 coloring from its companion package `tatva-coloring`.

We built `asdex` rather than contributing to the general-purpose packages above because none of them combined the "proper" sparsity detection paradigm (`jaxpr` tracing) with the state-of-the-art coloring techniques (distance-2 and star colorings).
Additionally, many of the alternatives listed above were made public only recently (we became aware of the concurrent works `JAX-AMG` and `tatva` as we were writing up this paper).

[^gh-sparsejac]: <https://github.com/mfschubert/sparsejac>
[^gh-sparsediffax]: <https://github.com/gdalle/sparsediffax>
[^gh-jax-nansparse]: <https://github.com/nardi/jax-nansparse>
[^gh-jax2sympy]: <https://github.com/johnviljoen/jax2sympy>
[^gh-jax-amg]: <https://github.com/jx-wang-s-group/JAX-AMG>
[^gh-tatva]: <https://github.com/smec-ethz/tatva>

# Software design

## Core features

`asdex` mirrors the main stages of ASD as distinct, composable components.

**Sparsity detection.**
`asdex` performs abstract interpretation on a `jaxpr`[^jaxpr] (short for *JAX expression*),
JAX's intermediate representation of a function.
Abstract interpretation propagates index sets through each `jaxpr` primitive to determine which inputs influence which outputs,
yielding a conservative pattern that may contain false positives but never misses a nonzero.
Working at the `jaxpr` level, the detector reuses JAX's own program representation
and naturally handles functions built from arbitrary JAX primitives.
It also makes second-order detection free:
because `jax.grad(f)` produces an ordinary JAX function with a `jaxpr` of its own,
and the Hessian of a scalar-valued function `f` is the Jacobian of its gradient,
`asdex` obtains Hessian patterns by applying the same first-order detector to `jax.grad(f)`,
with no separate second-order detector to implement or maintain.

[^jaxpr]: <https://docs.jax.dev/en/latest/jaxpr.html>

**Coloring.**
The detected pattern gives rise to a graph coloring problem, which `asdex` solves approximately using standard greedy algorithms.
For Jacobians, a distance-2 coloring of a bipartite row-column graph partitions columns (forward mode) or rows (reverse mode) into structurally orthogonal groups (with no overlapping nonzeros) [@gebremedhin2005what_color].
By default, `asdex` picks whichever mode requires fewer colors.
For Hessians, a star coloring exploits the symmetry of the matrix to further reduce the number of colors [@gebremedhin2007acyclic_star].
In both cases, the resulting groups are encoded in a seed matrix.

**Compressed differentiation and decompression.**
The compressed matrix stems from parallelized (`jax.vmap`-ed[^chunk-size]) Jacobian-vector or vector-Jacobian products, evaluated against the seed matrix.
All nonzero coefficients are obtained in this way and then efficiently scattered back into the original sparse format required for the Jacobian or Hessian.

[^chunk-size]: By specifying a `chunk_size` in `asdex`, the number of products computed in parallel can be limited via `jax.lax.map`, bounding memory usage.

## Performance and interface

**Preparation.**
A central design choice separates
one-time *preparation* (detection and coloring, which depend on input shapes but not values)
from repeated *evaluation* (compressed differentiation and decompression) of the sparse derivative.
In an iterative solver or a training loop, preparation is amortized, and only the cheap compressed AD pass is repeated,
echoing the preparation mechanism of `DifferentiationInterface.jl` [@dalle2025di_paper].
While detection and coloring are not JAX transformations themselves,
the prepared derivative function they produce is an ordinary JAX function,
so it composes with `jax.jit` and `jax.vmap` and runs on CPU, GPU, and TPU backends.

The API deliberately stays close to JAX's own and matches its semantics:
where a dense Jacobian is written as `jax.jacobian(f)`, `asdex.jacobian(f, x)` differs in only two aspects:

- it takes a sample input `x` for preparation
- it returns a function computing a sparse matrix (defaulting to JAX's `BCOO` format)

```python
import jax
import asdex

def f(x):
    return (x[1:] - x[:-1]) ** 2

x = jax.numpy.zeros(1000)
# Preparation: detect sparsity & color, return jit-able function
jac_fn = jax.jit(asdex.jacobian(f, x))
# Evaluation: cheap compressed AD passes and decompression
J = jac_fn(x)
```

Beyond this core, `asdex` mirrors the breadth of JAX's derivative API.
Like JAX's own transformations, it accepts arbitrary `PyTree` inputs and outputs rather than only flat vectors.
Multiple arguments (with the `argnums` keyword, enabling partial differentiation)
and auxiliary outputs (with the `has_aux` keyword) are supported as well.
Users can further supply known sparsity patterns, reuse colorings across sessions, limit their memory usage,
and decompress results into the dense and sparse formats of their choice, including those from `jax.experimental.sparse`, `numpy`, and `scipy.sparse`.
The package is developed openly under the MIT license.
Its documentation spans a tutorial, how-to guides, an API reference, a contributor's guide,
and continuously published benchmarks tracking detection, coloring, and evaluation performance.

# Research impact statement

`asdex` was inspired by JAX issue #1032[^jax-issue-1032], a feature request for sparse Jacobian and Hessian support, which has remained open since 2019.
This issue centralizes most of the related discussions and shows that there is sustained interest in such a feature in the JAX ecosystem.
As for `asdex` itself, it has already been
applied to accelerate the computation of Hessians in machine learning interatomic potentials [@langer2026truncated]
and has been integrated into `splineax`[^gh-splineax],
a sparsity-focused extension of the popular linear and least-squares library `lineax` [@rader2023lineax].

[^jax-issue-1032]: <https://github.com/jax-ml/jax/issues/1032>
[^gh-splineax]: <https://github.com/nardi/splineax>

# AI usage disclosure

Generative AI tools (Anthropic's Claude Opus 4.5 to 5.5 and Fable 5 models) were used in developing the `asdex` software and writing this manuscript.
All AI-generated code and text were reviewed and verified by the authors.
The design of `asdex` is based directly on prior work by the authors that was written without generative AI tools,
namely the peer-reviewed Julia packages
`DifferentiationInterface.jl` [@dalle2025di_paper],
`SparseConnectivityTracer.jl` [@hill2025sct_paper],
and `SparseMatrixColorings.jl` [@montoison2025revisiting_coloring],
from which it inherits its algorithmic approach.

# Acknowledgements

Adrian Hill gratefully acknowledges funding from the German Federal Ministry for Research, Technology and Space under the grant BIFOLD26B.
We thank Alexis Montoison for his work on the Julia ASD ecosystem,
Marcel Langer and Denis Korolev for their feedback on `asdex`,
as well as Jonathan Brodrick, John Viljoen and Johanna Hafner for productive discussions around JAX.

# References
