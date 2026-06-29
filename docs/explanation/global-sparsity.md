# Global Sparsity Patterns

Before detecting sparsity or coloring a matrix,
we need to decide _which_ sparsity pattern to work with.
This page explains the distinction between local and global patterns
and **why `asdex` uses global patterns exclusively**.

## Local vs. Global Patterns

Consider \(f(x) = x_1 \cdot x_2\), whose Jacobian is \([x_2,\; x_1]\).

**A local pattern** is the sparsity at a specific input point \(x\),
so it depends on the numerical values.
At \(x = (1, 0)\), the local pattern is `[0, 1]`,
but at \(x = (0, 1)\) it is `[1, 0]`.

**A global pattern** is the union of local patterns over the entire input domain,
making it input-independent.
For the same example, the global pattern is always `[1, 1]` — no sparsity.
Global patterns are supersets of local patterns:
less sparse, but valid everywhere.

## Why Conservative Patterns Are Safe

A sparsity pattern used for [coloring](coloring.md) must be either accurate or conservative.

If the pattern **misses a nonzero entry** (under-approximation),
the coloring may merge rows or columns that actually conflict,
silently producing **wrong results**.
There is no way to detect this error after the fact.

If the pattern **includes extra nonzeros** (over-approximation),
the coloring simply uses more colors than strictly necessary.
The computed Jacobian is still correct — just slightly less efficient to obtain.

This asymmetry is why `asdex` errs on the side of conservatism:
correctness comes first.
You can confirm a detected pattern against vanilla JAX.
See [Verifying Correctness](../how-to/verification.md).

## Why Global Over Local

[Sparsity detection](sparsity-detection.md) and [graph coloring](coloring.md) are preprocessing steps
that happen before any Jacobian is actually computed.
Both are expensive enough that we want their results to be **reusable**
across many evaluations at different input points.

A global pattern is input-independent by construction,
so the detection and coloring work is done once and amortized.
A local pattern would need to be recomputed every time the input changes,
negating most of the efficiency gains.

`asdex` achieves input-independence by using
[abstract interpretation](https://en.wikipedia.org/wiki/Abstract_interpretation)
rather than numerical evaluation.
Instead of plugging in concrete numbers,
it propagates index sets through the computation graph.
The result depends only on the function's structure,
not on any particular input.
The details are covered in [Sparsity Detection](sparsity-detection.md).

## The Trade-Off

Global patterns may be less sparse than local ones.
In the \(f(x) = x_1 \cdot x_2\) example,
the global pattern is fully dense even though every local pattern has a zero.
This means a few extra colors and a few extra AD passes.

In practice, the trade-off is overwhelmingly favorable:
the cost of extra colors is small,
while the ability to reuse a single coloring across all inputs
is what makes [automatic sparse differentiation](asd.md) practical.

!!! tip

    If a sparsity pattern looks overly conservative for your function,
    please help out `asdex`'s development by
    [reporting it](https://github.com/adrhill/asdex/issues).
    These reports directly drive improvements
    and are one of the most impactful ways to contribute.

## Precision Over Speed

`asdex` is designed around the assumption
that detection and coloring costs are amortized over many Jacobian evaluations.
Given this, it favors **sparser patterns** even if detecting them is slower:
fewer nonzeros lead to fewer colors,
and fewer colors mean fewer AD passes on every subsequent evaluation.
The one-time cost of a more precise analysis
is quickly repaid by cheaper evaluations.
