# Could sparsity detection be improved or sped up with `jax.linearize`?

Question: could Jacobian and/or Hessian sparsity detection be improved or sped up
using [`jax.linearize`](https://docs.jax.dev/en/latest/_autosummary/jax.linearize.html)?

All claims below were verified empirically against this checkout
(branch `ah/fable-detection-fixes`, JAX 0.10.2).
No code was changed.

## Executive summary

**As a speedup: no.**
Detection on the linearized computation is measurably slower,
1.5x to 1.9x for Jacobians and 1.2x to 1.3x for Hessians,
because the JVP jaxpr roughly doubles the number of equations
while the tangent half does the same index-set work as today.
There is no asymptotic win, and a structural floor argument (below) says parity is the best case.

**As a correctness and robustness improvement: yes, substantially.**
Detection on the linearized computation answers the question
"which inputs influence output *derivatives*"
instead of the current "which inputs influence output *values*".
The two coincide for most primitives, but not all,
and the mismatch is not hypothetical:

1. **Confirmed silent-wrong-Jacobian bug (new finding).**
   For a `custom_jvp` function whose derivative rule differs from the primal structure
   (a straight-through estimator),
   `asdex.jacobian` today returns an all-zero matrix while the true Jacobian is `2·I`.
   No error or warning is raised.
   Detection on the JVP jaxpr gets this right by construction.
   This is not covered by `detection-review.md` or the existing test suite.
2. **Coverage win.**
   Functions using unhandled special functions (e.g. `igamma`)
   raise `NotImplementedError` today
   but detect fine through the JVP jaxpr with zero new handlers,
   because value-only primitives fall off the derivative path.
3. **Precision wins.**
   Stop-gradient-style custom rules and integer-cast round trips
   currently over-report nonzeros that AD can never produce.

**What it does not fix:**
the structural handler bugs pinned in commit `aed9f14`
(`while` const ordering, scatter duplicate/clip semantics, integer `div` consts, bounds).
The JVP of a `while` is still a `while` and the JVP of a `scatter` is still a `scatter`,
so those handlers, and const and bounds tracking, are needed either way.
Landing those fixes stays necessary under any architecture.

**Recommendation:**
do not switch the architecture for speed.
Apply the linearize idea *surgically*:
fix the `custom_jvp`/`custom_vjp` soundness gap by propagating through the custom derivative rule instead of the primal `call_jaxpr`,
and add a cheap sound fallback for unknown primitives whose inputs carry no index sets.
Details in the last section.

## Background: what `jax.linearize` would mean here

`jax.linearize(f, x)` is `jax.jvp` plus partial evaluation:
it splits the JVP program into a primal part (evaluated at `x`, producing residuals)
and a tangent part that is *linear* in the tangent input.

Detection in asdex is shape-abstract (`make_jaxpr` on avals, no concrete point),
and under abstract tracing, `linearize` and `jvp` stage out identical jaxprs
(verified: same equation count on a test function).
So "detection via `jax.linearize`" concretely means:

> Build the jaxpr of `(x, v) ↦ jvp(f, (x,), (v,))[1]`,
> seed the tangent input `v` with identity index sets and the primal input `x` with empty sets,
> and propagate with the existing interpreter.

Why this computes the Jacobian pattern:
the tangent output is `J(x)·v`, linear in `v`,
so output element `i` depends on `v[j]` exactly when `J[i,j]` is structurally nonzero.
Because coefficients like `cos(x)` in the tangent program carry empty index sets,
they never contribute columns.
The pattern stays global and conservative (valid for all `x`),
since no values are folded during abstract tracing.

The decisive conceptual property:
this is *the same program that AD actually executes* during decompression.
Detection and differentiation can no longer disagree about custom rules,
zero-derivative ops, or `stop_gradient`.
The current primal-jaxpr analysis has to *re-encode* JAX's derivative semantics by hand
(the zero-derivative case list, `_prop_clamp`, comparison handlers, and so on),
and every re-encoding is a chance to diverge from AD.

The Hessian analog seeds `v` in `(x, v) ↦ jvp(grad(f), (x,), (v,))[1]`,
which is exactly the fwd-over-rev HVP that decompression uses.

## The approach needs no interpreter changes to prototype

The whole scheme is expressible with today's public API,
because `argnums` seeding already implements "empty sets on non-selected inputs":

```python
def jvp_wrap(f):
    return lambda x, v: jax.jvp(f, (x,), (v,))[1]

# current:    jacobian_sparsity(f, x)
# linearized: jacobian_sparsity(jvp_wrap(f), x, x, argnums=1)
```

All experiments below use this wrapper.

## Experiment 1: pattern equivalence

A 15-function battery (elementwise, stencil, `floor`, `where`, gather, scatter-add,
`dot_general`, `cumsum`, `cond`, `while`, `scan`, reductions, conv, two `custom_jvp` variants),
each checked against a dense `jax.jacfwd` reference.

Result: on 13 of 15 functions the two approaches produce **identical** patterns,
and both are sound supersets of the reference.
Control flow (`while`, `cond`, `scan`) traces fine under `jvp`,
and the resulting doubled-carry loops go through the existing handlers unchanged.

The two differing cases are both `custom_jvp`, analyzed next.

## Experiment 2: where the linearized view is strictly better

### 2a. `custom_jvp` with a derivative rule that adds dependencies (unsound today)

Straight-through estimator, the standard trick for quantized/discrete layers:

```python
@jax.custom_jvp
def ste_round(x):
    return jnp.round(x)

@ste_round.defjvp
def ste_round_jvp(primals, tangents):
    (x,), (t,) = primals, tangents
    return jnp.round(x), t          # derivative rule: identity

def f(x):
    return ste_round(x) * 2.0
```

| | result |
|---|---|
| `jax.jacfwd(f)(x)` | `2·I` (3 nnz) |
| `jacobian_sparsity(f, x)` | **empty pattern (0 nnz)** |
| `asdex.jacobian(f, x)(x)` | **all-zero matrix, no error** |
| JVP-based detection | diagonal (3 nnz), correct |

The cause is `_prop_dispatch` handling `custom_jvp_call` by tracing the *primal* `call_jaxpr`
(`_interpret/__init__.py:295`).
The primal contains `round`, which the interpreter correctly classifies as zero-derivative,
but the user's custom rule overrides that derivative, and detection never sees the rule.
This produces missing nonzeros, so decompression silently returns a wrong Jacobian,
the failure class the project's design philosophy explicitly rules out.

The existing test `test_custom_jvp_closure_captured_index` only covers a custom rule
whose structure matches the primal, so it cannot catch this.
`detection-review.md` does not cover it either.
Worth pinning with a `bug`-marked test per `tests/CLAUDE.md` regardless of the fix chosen.

### 2b. `custom_jvp` that removes dependencies (imprecise today)

A stop-gradient-flavored rule (primal `x * 1.0`, JVP rule returns zero tangent)
detects 5 nnz today versus the true 3 nnz.
Sound but wasteful: extra nonzeros mean extra colors.
JVP-based detection is exact.

### 2c. Integer-cast round trips (imprecise today)

```python
def f(x):
    return jnp.sin(x.astype(jnp.int32).astype(jnp.float64))
```

AD gives an identically zero Jacobian (integer casts are non-differentiable, tangents die).
Primal-based detection reports a dense diagonal (3 nnz), JVP-based reports 0 nnz.

### 2d. Unhandled primitives off the derivative path (hard error today)

```python
def f(x):
    return jax.scipy.special.gammainc(2.0, x)   # lowers to igamma
```

Primal-based detection raises `NotImplementedError: No handler for primitive 'igamma'`.
JVP-based detection works *today*, with zero new handlers.
Mechanism: the derivative of `igamma` w.r.t. `x` is `exp(-x + (a-1)·log x - lgamma(a))`,
so the tangent path only contains `exp`/`log`/`lgamma`/`mul`,
and `_dce_closed_jaxpr` removes the primal `igamma` equation
because only the tangent output is returned.

Caveat: DCE alone is not a complete coverage story.
When a primitive's JVP rule reuses the primal output (as `tanh` does),
the primal equation survives DCE and an unknown primitive would still throw.
A one-line-of-concept fallback closes this completely and is sound in general:
if every input of an unknown primitive carries only empty index sets,
its outputs get empty index sets
(dependence can only flow through `invars`, and consts are seeded empty).
Under JVP-based detection this covers *every* value-only primitive automatically,
and JAX guarantees the primitives that do carry tangent flow are linear ops
that the interpreter already handles well.

## Experiment 3: what it does not fix

The bug classes pinned in `aed9f14` are structural, not derivative-semantic:

- `while` binds `[cond_consts, body_consts, carry]` (C1): the JVP of a `while` is a `while` with a doubled carry, same handler, same bug.
- Integer `div`/`rem` const semantics (C2), scatter duplicate/clip semantics (C3, C4), bounds through `div` (C6): const and bounds tracking runs on the primal half of the JVP jaxpr exactly as it does today, because gather/scatter index computations are primal values.
- Conservative-fallback gaps (G1, G2, G3): unchanged.

So the linearize idea shrinks the *derivative-classification* surface
(the zero-derivative lists, custom rules, comparisons)
but not the *structural* surface (index mapping, consts, bounds),
which is where the currently pinned bugs live.

## Experiment 4: performance

Jaxpr sizes (equations, nested jaxprs included):

| function | primal | JVP | ratio |
|---|---|---|---|
| MLP (2 layers) | 8 | 18 | 2.3x |
| Brusselator RHS | 36 | 73 | 2.0x |
| stencil | 9 | 24 | 2.7x |

Detection wall time (best of 3):

| case | primal-based | JVP-based | ratio |
|---|---|---|---|
| Jacobian, Brusselator, n=1 000 | 1.7 ms | 3.3 ms | 1.92x |
| Jacobian, Brusselator, n=10 000 | 16.2 ms | 24.0 ms | 1.48x |
| Hessian, Rosenbrock, n=1 000 | 4.7 ms (grad-based) | 6.2 ms (HVP-based) | 1.33x |
| Hessian, Rosenbrock, n=10 000 | 40.5 ms | 47.3 ms | 1.17x |

(Hessian patterns were verified identical and sound in both approaches.)

Why there is no speedup to be had here:
the tangent half of the JVP jaxpr mirrors the primal computation essentially one-to-one,
so it performs the same set-union work the primal analysis performs today,
plus the interpreter walks the primal half (cheap, empty sets, but not free),
plus tracing time roughly doubles.
Skipping empty-set equations would narrow the gap toward parity, not below it.
Real detection speedups would come from the index-set backend
(the `_common.py` factory helpers exist precisely to allow swapping `set[int]` for bitsets or numpy),
which is orthogonal to what the jaxpr being analyzed looks like.

One theoretical exception:
functions with large value-only subcomputations feeding zero-derivative sinks
get those subgraphs DCE'd entirely under the JVP view.
This is marginal in practice.

## Limitations of a wholesale switch

- **`custom_vjp` functions stage to `custom_lin`.**
  Under abstract tracing, `jvp` of a `custom_vjp` function does not error
  (as it would eagerly) but emits a `custom_lin` primitive,
  which has no handler today.
  A conservative handler is trivial and sound.
  A precise one must trace the user's `bwd` function and transpose the resulting pattern.
- **Non-differentiable inputs.**
  Selected `argnums` with integer or boolean dtypes have no float tangents,
  so the wrapper construction needs care
  (AD semantics would report empty rows/columns for them, which is arguably the right answer).
- **Global validity is preserved, but a local mode becomes possible.**
  Abstract seeding keeps the "valid for all inputs" guarantee.
  Running `linearize` at a *concrete* point instead would feed concrete residuals
  into `state_consts` and produce tighter, input-specific patterns
  (the analog of SparseConnectivityTracer.jl's `TracerLocalSparsityDetector`).
  That is incompatible with reusing a coloring across inputs,
  so it could only ever be a clearly-labeled opt-in feature, not a replacement.

## Recommendations

1. **Do not rewrite detection on top of `jax.linearize` for performance.**
   It is a 1.2x to 1.9x slowdown with no asymptotic upside.

2. **Fix the `custom_jvp`/`custom_vjp` soundness gap now, using the linearize idea locally.**
   For `custom_jvp_call`, propagate through the custom JVP rule
   (seed primal slots empty, tangent slots with the equation's input index sets,
   read the tangent outputs) instead of the primal `call_jaxpr`.
   The rule is stored as a `lu.WrappedFun` in `eqn.params["jvp_jaxpr_fun"]`,
   so materializing its jaxpr needs internal API,
   with a conservative fallback if that fails.
   If tracing the rule is deemed too fragile,
   the honest interim behavior is a conservative fallback or a hard error for `custom_jvp_call`/`custom_vjp_call`,
   never the current silent primal trace
   ("favor exceptions over wrong results").
   Either way, pin the straight-through-estimator case with a `bug`-marked test first.

3. **Add the empty-inputs shortcut to the fallback branch of `_prop_dispatch`.**
   Before `_prop_throw_error`, if all input index sets of an unknown primitive are empty,
   emit empty output index sets instead of raising.
   This is sound unconditionally,
   keeps all existing precise handlers (and their const/bounds tracking) untouched,
   and already pays off today whenever `argnums` deselects the inputs
   feeding an exotic primitive.
   Downstream consumers of the skipped output see missing consts/bounds
   and fall back conservatively per the documented invariant.

4. **Document the JVP wrapper as a cross-check recipe.**
   `jacobian_sparsity(lambda x, v: jax.jvp(f, (x,), (v,))[1], x, x, argnums=1)`
   works with the public API today and detects
   exactly what AD will compute,
   which makes it a useful validation tool for codebases heavy in custom derivative rules,
   and a candidate extra check inside `check_jacobian_correctness`.

5. **Optionally revisit a `linearized` detection mode long-term.**
   If custom-rule-heavy ecosystems (Equinox, Diffrax style libraries)
   keep producing detection/AD mismatches,
   aligning detection with AD semantics by construction may become worth the ~1.5x detection cost.
   The prototype path is already proven by the wrapper trick.

## Reproduction

Experiment scripts (battery, sizes, timings, edge cases) were run from the session scratchpad.
The key reproductions are inlined above and depend only on `asdex` and JAX 0.10.2.
