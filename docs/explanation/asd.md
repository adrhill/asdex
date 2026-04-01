# Automatic Sparse Differentiation

Automatic sparse differentiation (ASD) exploits the sparsity structure of Jacobians and Hessians
to compute them far more efficiently than dense automatic differentiation (AD).
This page explains the mathematical ideas behind the approach.

## Automatic Differentiation in Brief

AD computes exact derivatives of programs
by applying the chain rule to each elementary operation.
For a function \(f: \mathbb{R}^n \to \mathbb{R}^m\),
AD can evaluate Jacobian-vector products in two modes:

- **Reverse mode** computes a **vector-Jacobian product** (VJP): given a covector \(v \in \mathbb{R}^m\),
  it returns \(v^\top J\) in roughly the same time as one evaluation of \(f\).
  Setting \(v = e_i\) extracts row \(i\) of the Jacobian.
- **Forward mode** computes a **Jacobian-vector product** (JVP): given a tangent vector \(t \in \mathbb{R}^n\),
  it returns \(Jt\) in roughly the same time as one evaluation of \(f\).
  Setting \(t = e_j\) extracts column \(j\) of the Jacobian.

In both cases, one AD pass produces a single row or column —
not the full Jacobian.
Building the complete matrix requires multiple passes,
and reducing their number is the goal of sparse differentiation.

## Computational Cost of AD

Consider a function \(f: \mathbb{R}^n \to \mathbb{R}^m\) with Jacobian \(J \in \mathbb{R}^{m \times n}\).
Computing \(J\) by standard autodiff requires either:

- \(m\) VJPs (reverse-mode): one per row, or
- \(n\) JVPs (forward-mode): one per column.

For large, sparse Jacobians — common in scientific computing —
most entries of \(J\) are zero,
yet dense autodiff pays the full cost of \(m\) or \(n\) passes.
ASD reduces this to a number of passes proportional to the
number of colors in a graph coloring of the sparsity pattern,
which is often orders of magnitude smaller.

## Structural Orthogonality

The key insight is structural orthogonality:
two rows \(i_1\) and \(i_2\) of \(J\) are structurally orthogonal
when they share no nonzero column —
that is, there is no column \(j\) where both \(J_{i_1 j}\) and \(J_{i_2 j}\) are (potentially) nonzero.

If two rows are structurally orthogonal,
their VJPs can be combined into a single pass
by summing their seed vectors.
The nonzeros don't overlap,
so the result is unambiguous.
The same idea applies symmetrically to columns and JVPs.

## Seed Matrices and Compression

Graph coloring assigns each row a color
such that rows sharing a nonzero column get different colors.
From this coloring, we build a seed matrix \(S \in \mathbb{R}^{m \times c}\),
where \(c\) is the number of colors.
The seed for color \(c\) is the sum of the standard basis vectors
for all rows assigned that color:

\[
S_{:,c} = \sum_{i \,:\, \text{color}(i) = c} e_i
\]

The compressed Jacobian is then:

\[
B = S^\top \cdot J \in \mathbb{R}^{c \times n}
\]

Computing \(B\) requires only \(c\) VJPs — one per color — instead of \(m\).

## Decompression

Recovering the sparse Jacobian from \(B\) is called decompression.
Because same-colored rows are structurally orthogonal,
each nonzero \(J_{ij}\) appears in exactly one entry of \(B\):

\[
J_{ij} = B_{\text{color}(i),\, j}
\]

We simply read off each nonzero from the compressed matrix
using the known color assignments.
This is **direct decompression** — no systems of equations need to be solved.

## The Three-Step Pipeline

ASD decomposes the problem into three independent stages:

1. **[Detection](sparsity-detection.md)** — determine the [global sparsity pattern](global-sparsity.md) of the Jacobian
   by analyzing the computation graph (no numerical evaluation).
2. **[Coloring](coloring.md)** — assign colors to rows or columns
   so that structurally orthogonal groups share a color.
3. **Decompression** — compute one AD pass per color and extract the sparse matrix.

Steps 1 and 2 are preprocessing:
they depend only on the function's structure and input shape,
not on the input values.
Once computed, the coloring can be reused across arbitrarily many evaluations.
Step 3 is the only part that touches actual numerical data.

## Amortization

The three-step split is designed around amortization.
Detection and coloring are the most expensive steps,
but their results depend only on the function's structure and input shape —
not on the input values.
This means they can be computed once and reused
across arbitrarily many evaluations at different inputs.

In a typical workflow,
a user calls [`jacobian_coloring`](../reference/index.md#asdex.jacobian_coloring) (or [`hessian_coloring`](../reference/index.md#asdex.hessian_coloring)) once during setup
and passes the result to [`jacobian_from_coloring`](../reference/index.md#asdex.jacobian_from_coloring) (or [`hessian_from_coloring`](../reference/index.md#asdex.hessian_from_coloring)) in a loop.
The per-evaluation cost is then just the decompression step:
\(c\) AD passes plus a cheap index lookup,
where \(c\) is the number of colors.
For problems where the Jacobian is evaluated hundreds or thousands of times —
such as implicit solvers, optimization, or time-stepping —
the preprocessing cost becomes negligible.

This amortization assumption also guides design decisions in `asdex`:
it is worth spending more time on [detection](sparsity-detection.md)
if it produces [sparser patterns](global-sparsity.md),
because fewer nonzeros lead to fewer colors
and fewer AD passes on every subsequent evaluation.

## Extension to Hessians

For a scalar function \(f: \mathbb{R}^n \to \mathbb{R}\),
the Hessian \(H \in \mathbb{R}^{n \times n}\) is symmetric: \(H_{ij} = H_{ji}\).

Since the Hessian is the Jacobian of the gradient,
sparsity detection reduces to Jacobian detection:
\(\operatorname{hessian\_sparsity}(f) = \operatorname{jacobian\_sparsity}(\nabla f)\).
This works because `jax.grad` produces a jaxpr like any other function,
so the same [interpreter](sparsity-detection.md) handles both cases with no extra machinery.
And because the Hessian is symmetric,
coloring can exploit this via [star coloring](coloring.md),
which typically needs far fewer colors than treating the Hessian as a general matrix.

Each color corresponds to one Hessian-vector product (HVP),
computed via forward-over-reverse autodiff.
The decompression step recovers both \(H_{ij}\) and \(H_{ji}\) from each entry,
halving the effective number of unknowns.

## References

- [_An Illustrated Guide to Automatic Sparse Differentiation_](https://iclr-blogposts.github.io/2025/blog/sparse-autodiff/), Hill, Dalle, Montoison (2025) — a visual walkthrough of the ideas on this page.
