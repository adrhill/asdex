# Plan: Document all features in README + doc pages (issue #152)

Tracking issue [#152](https://github.com/adrhill/asdex/issues/152)
(*docs: list all features in README, linking to relevant doc pages*),
milestone **JOSS**, with subissues:

| Issue | Title | Source feature |
|-------|-------|----------------|
| [#148](https://github.com/adrhill/asdex/issues/148) | docs: document new decompression functionality | output formats ([#142]) |
| [#149](https://github.com/adrhill/asdex/issues/149) | docs: document chunked evaluation | `chunk_size` ([#138]/[#139]) |
| [#150](https://github.com/adrhill/asdex/issues/150) | docs: document arbitrary PyTree inputs/outputs | `argnums` / PyTrees ([#82]/[#105]) |
| [#151](https://github.com/adrhill/asdex/issues/151) | docs: document auxiliary outputs | `has_aux` ([#105]) |

The issue-#152 comment also asks: *"add a page on the methods verifying correctness."*
This plan adds a dedicated **`how-to/verification.md`** page for that, and trims each
guide's *Verifying Results* section to a short pointer to it.

[#142]: https://github.com/adrhill/asdex/pull/142
[#139]: https://github.com/adrhill/asdex/pull/139
[#138]: https://github.com/adrhill/asdex/issues/138
[#105]: https://github.com/adrhill/asdex/pull/105
[#82]: https://github.com/adrhill/asdex/issues/82
[#143]: https://github.com/adrhill/asdex/pull/143

## Scope & approach

These features already ship in `v0.4.0`; the work is pure **documentation**, no source
changes. The four documented features (output formats, PyTrees/multi-arg, aux, chunking)
are **cross-cutting**: they apply to both Jacobians *and* Hessians.

**The four features fold into the two existing guides; verification gets its own page.**
Each cross-cutting feature becomes a dedicated section in both `how-to/jacobians.md` and
`how-to/hessians.md`, which are restructured to **mirror each other** section-for-section,
so a reader who knows the Jacobian guide finds the Hessian guide laid out identically.
Correctness checking (which already lives in both guides as a near-duplicate *Verifying
Results* section and is a task in its own right) is lifted into a single new page,
`how-to/verification.md`; each guide keeps a short pointer to it.

This keeps each feature next to the API it modifies (so the examples use the real
`jacobian`/`hessian` calls) while giving verification the standalone page the issue-#152
comment asks for. Nav grows by exactly one entry.

**Every code block runs live during the build.** All new snippets carry `markdown-exec`'s
`exec="true" source="above"`, so `mkdocs` executes them and renders their real output, with no
hardcoded results. Output stays small and deterministic (types, shapes, nnz), matching the
existing convention of `print(f"```\n{coloring}\n```")`. Every snippet below was also run
standalone against the working tree (`v0.4.0`) and produces the stated result, so the
build cannot regress silently.

**All prose uses semantic line breaks.** Per `docs/CLAUDE.md` (§ *Semantic Line Breaks*,
a firm requirement), every sentence or clause sits on its own line. Break after each
period, and after commas in long sentences. All new and edited prose below already follows
this; reviewers should preserve it rather than reflowing to a fixed width.

Branch: `docs/feature-overview-152` (per the feature-branch convention).

---

## Updated guide outlines (mirrored)

Each guide is reorganized into two tiers,
so importance shows up both in the page and in its table of contents:

- a **`## Basics`** section with the sections every reader needs:
  basic usage and primal returns,
  the precompute / save-load / known-pattern workflow that avoids paying the expensive
  detection-and-coloring step on every call,
  and a pointer to verification.
- a **`## Advanced`** section for situational options:
  mode selection, the lower-level pipeline, and the four newly documented features.

Every existing section keeps its heading text and is demoted from `##` to `###` under one
of the two tiers.
MkDocs slugifies heading *text*, not level,
so all in-page anchors stay valid,
including the README and `index.md` links that target them.

The two guides mirror section-for-section.
The only intentional divergence is mode-selection
(Jacobians choose row vs column coloring;
Hessians have symmetric coloring plus an HVP-mode choice)
and that **PyTree *outputs*** appear only in the Jacobian guide
(a Hessian requires a scalar output).
New sections are marked **NEW**.

| Tier | `how-to/jacobians.md` | `how-to/hessians.md` |
|------|------------------------|----------------------|
| *(intro)* | Intro + *"verify once"* tip | Intro + *"verify once"* tip |
| **`## Basics`** | `### Basic Usage` | `### Basic Usage` |
| | `### Getting the Primal Value Too` | `### Getting the Primal Value Too` |
| | `### Precomputing the Colored Pattern` | `### Precomputing the Colored Pattern` |
| | `### Saving and Loading Patterns` | `### Saving and Loading Patterns` |
| | `### Manually Providing a Sparsity Pattern` | `### Manually Providing a Sparsity Pattern` |
| | `### Verifying Results` → pointer | `### Verifying Results` → pointer |
| **`## Advanced`** | `### Choosing Row vs Column Coloring` | `### Symmetric Coloring` |
| | | `### Choosing an HVP Mode` |
| | `### Separate Detection and Coloring` | `### Separate Detection and Coloring` |
| | `### Multiple Inputs and Outputs` **(NEW, #150)** | `### Multiple Inputs` **(NEW, #150)** |
| | `### PyTree Inputs and Outputs` **(NEW, #150)** | `### PyTree Inputs` **(NEW, #150)** |
| | `### Auxiliary Outputs` **(NEW, #151)** | `### Auxiliary Outputs` **(NEW, #151)** |
| | `### Output Formats` **(NEW, #148)** | `### Output Formats` **(NEW, #148)** |
| | `### Reducing Peak Memory with Chunking` **(NEW, #149)** | `### Reducing Peak Memory with Chunking` **(NEW, #149)** |

Plus one **new page**, `how-to/verification.md`, holding the consolidated correctness-checking how-to.

Within **`## Advanced`**, the existing sections that move up lead the tier
(*Choosing Row vs Column Coloring* / *Symmetric Coloring* + *Choosing an HVP Mode*,
then *Separate Detection and Coloring*),
and the five **NEW** feature sections are appended after them.
This keeps the new content as one contiguous block at the end of each guide,
which the two-commit strategy below relies on.

Changes vs. today:

- Add the two tier headings `## Basics` and `## Advanced`,
  and demote every existing `##` section to `###` under the right tier
  (heading text unchanged).
- Move *Choosing Row vs Column Coloring* (Jacobian),
  *Symmetric Coloring* + *Choosing an HVP Mode* (Hessian),
  and *Separate Detection and Coloring* to the top of *Advanced*.
- Drop the lone line in the Jacobian guide's *Basic Usage*
  (*"`asdex` supports multi-dimensional input and output arrays"*),
  since the new *Multiple Inputs and Outputs* section now covers it.
- Lift the two near-identical *Verifying Results* sections into `verification.md`;
  each guide's *Verifying Results* becomes a 2–3 line pointer,
  and the top-of-page *"verify once"* tip retargets to the new page.
- Retarget **every other inbound verification link** across the docs to the new page: the
  *Verifying Results* link in `tutorials/getting-started.md`, and a new pointer from the
  *"correctness comes first"* paragraph in `explanation/global-sparsity.md`. The full set
  of inbound links is enumerated under
  [Inbound links to `verification.md`](#inbound-links-to-verificationmd) below.

---

## New sections to add (verified snippets)

Each subsection below is appended to the **`## Advanced`** section of **both** guides,
as a `###` heading
(the snippets render the heading as `##` for readability —
create it as `###` so it nests under *Advanced*).
The prose mirrors;
only the example call differs (Jacobian vs Hessian).
All snippets are verified.

### Section 4: Multiple Inputs, and PyTree Inputs/Outputs (#150)

This material is split into two sibling `###` sections under *Advanced*.
The first covers multiple positional arguments and multiple outputs.
The second covers arbitrary PyTree containers.

#### Multiple Inputs and Outputs

**Jacobian guide** (`## Multiple Inputs and Outputs`):

````markdown
## Multiple Inputs and Outputs

`asdex` mirrors `jax.jacobian`:
it differentiates functions of several arguments,
selecting which arguments to differentiate with `argnums`.

Pass a sample value for each positional argument,
and select the ones to differentiate with `argnums`:

```python exec="true" session="jac-multi" source="above"
import jax
import jax.numpy as jnp
from asdex import jacobian

def f(x, y):
    return x * y

x = jnp.arange(1.0, 4.0)
y = jnp.arange(4.0, 7.0)

Jx, Jy = jax.jit(jacobian(f, x, y, argnums=(0, 1)))(x, y)  # one block per selected arg
```

```python exec="true" session="jac-multi"
print(f"""```
Jx: {type(Jx).__name__} {Jx.shape}
Jy: {type(Jy).__name__} {Jy.shape}
```""")
```

With an integer `argnums` (the default `0`) a single block is returned, not a tuple.
Arguments not named by `argnums` are still passed at call time and held fixed,
yet they can still influence the result.
Here `scale` is not differentiated, but it scales every entry of the Jacobian:

```python exec="true" session="jac-multi" source="above"
def scaled(x, scale):
    return scale * x ** 2

J2 = jacobian(scaled, x, 2.0, argnums=0)(x, 2.0)
J5 = jacobian(scaled, x, 5.0, argnums=0)(x, 5.0)
```

```python exec="true" session="jac-multi"
print(f"""```
scale=2 -> diagonal {J2.todense().diagonal()}
scale=5 -> diagonal {J5.todense().diagonal()}
```""")
```

A function may also return several outputs.
The Jacobian then mirrors the output structure,
with one block per (output, selected argument) pair, exactly like `jax.jacobian`:

```python exec="true" session="jac-multi" source="above"
def f_multi(x, y):
    return x * y, x + y  # two outputs

J = jax.jit(jacobian(f_multi, x, y, argnums=(0, 1)))(x, y)
```

```python exec="true" session="jac-multi"
dxy_dx = J[0][0]  # d(x * y) / dx: first output w.r.t. first argument
print(f"""```
outer length (outputs):    {len(J)}
inner length (arguments):  {len(J[0])}
J[0][0] = d(x*y)/dx:       {dxy_dx.shape}
```""")
```
````

**Hessian guide** (`## Multiple Inputs`): same opening prose (a Hessian requires a scalar
output, so there is no multiple-output case).
With a tuple `argnums` the result is a nested `(input_tree, input_tree)` grid,
mirroring `jax.hessian`:
`H[i][j]` is the second derivative with respect to argument `i` and argument `j`,
so the full block grid is shown rather than a single corner.

````markdown
## Multiple Inputs

```python exec="true" session="hess-multi" source="above"
import jax
import jax.numpy as jnp
from asdex import hessian

def f(x, y):
    return jnp.sum(x ** 2 * y)

x = jnp.arange(1.0, 4.0)
y = jnp.arange(4.0, 7.0)

H = jax.jit(hessian(f, x, y, argnums=(0, 1)))(x, y)
```

```python exec="true" session="hess-multi"
Hxx = H[0][0]  # ∂²f/∂x²
Hxy = H[0][1]  # ∂²f/∂x∂y
Hyy = H[1][1]  # ∂²f/∂y²
print(f"""```
grid shape:           {len(H)} x {len(H[0])}
H[0][0]  d2f/dx2:     {type(Hxx).__name__} {Hxx.shape}
H[0][1]  d2f/dx dy:   {type(Hxy).__name__} {Hxy.shape}
H[1][1]  d2f/dy2:     {type(Hyy).__name__} {Hyy.shape}
```""")
```
````

#### PyTree Inputs and Outputs

**Jacobian guide** (`## PyTree Inputs and Outputs`):

````markdown
## PyTree Inputs and Outputs

A single argument can itself be an arbitrary [PyTree](https://docs.jax.dev/en/latest/pytrees.html),
such as a dictionary of parameters.
The Jacobian comes back as a matching PyTree of blocks:

```python exec="true" session="jac-pt" source="above"
import jax
import jax.numpy as jnp
from asdex import jacobian

def loss(params):
    return params["weight"] * jnp.sin(params["bias"])

params = {"weight": jnp.arange(1.0, 4.0), "bias": jnp.linspace(0.0, 1.0, 3)}
J = jax.jit(jacobian(loss, params))(params)
```

```python exec="true" session="jac-pt"
print(f"""```
keys:        {sorted(J)}
J['weight']: {type(J['weight']).__name__} {J['weight'].shape}
J['bias']:   {type(J['bias']).__name__} {J['bias'].shape}
```""")
```

PyTree *outputs* are supported too.
The result has `(output_tree, input_tree)` structure, exactly like `jax.jacobian`:
one block per output leaf, each shaped `(*output_leaf_shape, *input_leaf_shape)`.

```python exec="true" session="jac-pt" source="above"
def f_out(x):
    return {"squared": x ** 2, "total": jnp.sum(x)}

x = jnp.arange(1.0, 4.0)
J = jax.jit(jacobian(f_out, x))(x)
```

```python exec="true" session="jac-pt"
print(f"""```
keys:         {sorted(J)}
J['squared']: {J['squared'].shape}   # (3,) output, (3,) input
J['total']:   {J['total'].shape}      # scalar output, (3,) input
```""")
```
````

**Hessian guide** (`## PyTree Inputs`): same opening prose, minus the PyTree-*output*
paragraph (a Hessian requires a scalar output).
For a PyTree argument the Hessian is a matching nested structure of blocks, here a
dict-of-dicts, where `H[i][j]` couples leaves `i` and `j`:

````markdown
## PyTree Inputs

```python exec="true" session="hess-pt" source="above"
import jax
import jax.numpy as jnp
from asdex import hessian

def loss(params):
    return jnp.sum(params["a"] ** 2 * params["b"])

params = {"a": jnp.arange(1.0, 4.0), "b": jnp.arange(4.0, 7.0)}
H = jax.jit(hessian(loss, params))(params)
```

```python exec="true" session="hess-pt"
Has = H["a"]["a"]  # ∂²/∂a²
Hab = H["a"]["b"]  # ∂²/∂a∂b
print(f"""```
outer keys:            {sorted(H)}
H['a']['a']  d2/da2:   {type(Has).__name__} {Has.shape}
H['a']['b']  d2/da db: {type(Hab).__name__} {Hab.shape}
```""")
```
````

### Section 5: Auxiliary Outputs (#151)

**Jacobian guide** (`## Auxiliary Outputs`):

````markdown
## Auxiliary Outputs

Set `has_aux=True` when your function returns `(output, auxiliary_data)`, mirroring `jax.jacrev`.
The auxiliary data is passed through untouched, useful for diagnostics, intermediate values, or model state.

```python exec="true" session="jac-aux" source="above"
import jax
import jax.numpy as jnp
from asdex import jacobian

def f(x):
    y = x ** 2
    return y, {"mean_sq": jnp.mean(y)}  # (output, aux)

x = jnp.arange(1.0, 4.0)
J, aux = jax.jit(jacobian(f, x, has_aux=True))(x)
```

```python exec="true" session="jac-aux"
print(f"""```
J:       {type(J).__name__} {J.shape}
mean_sq: {float(aux['mean_sq']):.3f}
```""")
```

[`value_and_jacobian`](../reference/index.md#asdex.value_and_jacobian) nests aux next to the value,
matching `jax.value_and_grad` ordering, giving `((value, aux), J)`:

```python exec="true" session="jac-aux" source="above"
from asdex import value_and_jacobian

(value, aux), J = value_and_jacobian(f, x, has_aux=True)(x)
```

```python exec="true" session="jac-aux"
print(f"""```
value:   {value.shape}
mean_sq: {float(aux['mean_sq']):.3f}
```""")
```


The auxiliary data may hold arbitrary Python objects, not just JAX arrays.
It is extracted from the forward pass that AD already runs,
so returning it adds no extra evaluation of `f`.
````

**Hessian guide** (`## Auxiliary Outputs`): same prose with `hessian` /
`value_and_hessian`:

````markdown
```python exec="true" session="hess-aux" source="above"
import jax
import jax.numpy as jnp
from asdex import hessian

def g(x):
    return jnp.sum(x ** 3), {"norm": jnp.linalg.norm(x)}  # (output, aux)

x = jnp.arange(1.0, 4.0)
H, aux = jax.jit(hessian(g, x, has_aux=True))(x)
```

```python exec="true" session="hess-aux"
print(f"""```
H:    {type(H).__name__} {H.shape}
norm: {float(aux['norm']):.3f}
```""")
```
````

### Section 6: Output Formats (#148)

**Jacobian guide** (`## Output Formats`):

````markdown
## Output Formats

By default, `asdex` returns sparse matrices as JAX [BCOO](https://docs.jax.dev/en/latest/jax.experimental.sparse.html) arrays.
The `output_format` argument selects a different container.
It is accepted by [`jacobian`](../reference/index.md#asdex.jacobian),
its `value_and_*` variant, and the `*_from_coloring` variants.

| `output_format` | Returned type | JIT-able by caller |
|-----------------|---------------|--------------------|
| `"bcoo"` (default) | `jax.experimental.sparse.BCOO` | yes |
| `"dense"` | `jax.Array` | yes |
| `"numpy_dense"` | `numpy.ndarray` | no |
| `"scipy_coo"` | `scipy.sparse.coo_array` | no |
| `"scipy_csr"` | `scipy.sparse.csr_array` | no |
| `"scipy_csc"` | `scipy.sparse.csc_array` | no |

```python exec="true" session="jac-fmt" source="above"
import jax
import jax.numpy as jnp
from asdex import jacobian

def f(x):
    return (x[1:] - x[:-1]) ** 2

x = jnp.arange(1.0, 6.0)

J_bcoo = jax.jit(jacobian(f, x))(x)                          # BCOO (default)
J_dense = jax.jit(jacobian(f, x, output_format="dense"))(x)  # jax.Array
J_csr = jacobian(f, x, output_format="scipy_csr")(x)         # scipy.sparse.csr_array
```

```python exec="true" session="jac-fmt"
print(f"""```
bcoo:       {type(J_bcoo).__name__}
dense:      {type(J_dense).__name__}  shape={J_dense.shape}
scipy_csr:  {type(J_csr).__name__}  nnz={J_csr.nnz}
```""")
```


!!! warning "Host formats are not JIT-able by the caller"

    `"numpy_dense"` and the scipy formats produce non-JAX arrays,
    so you cannot wrap the returned function in `jax.jit`.
    `asdex` JIT-compiles their core internally, so they stay fast anyway.
    Just call them directly:

    ```python
    J = jacobian(f, x, output_format="numpy_dense")(x)  # do NOT jax.jit this
    ```

!!! info "SciPy formats are 2D-only"

    SciPy sparse arrays are strictly 2D.
    They require the input and output to each be a single flat 1D array.
    `asdex` flattens and checks the full input structure up front.
    Any other shape, such as a multi-dimensional array, multiple arguments,
    or an arbitrarily nested PyTree, raises a clear `ValueError` rather than a wrong result.
    Note that SciPy is an optional dependency. Install it via `pip install 'asdex[scipy]'`.

Structural non-zeros that happen to be numerically zero at the evaluation point are kept as explicit entries in the `BCOO` and scipy outputs,
so the structure always matches the detected [global sparsity pattern](../explanation/global-sparsity.md) and is independent of `x`.
````

**Hessian guide** (`## Output Formats`): same table and callouts (the SciPy note drops the
"output" clause, since only the input must be a flat 1D array), with a `hessian` example.
The same up-front structure check applies,
so a multi-dimensional, multi-argument, or nested-PyTree input raises the same clear
`ValueError` rather than silently producing a wrong result:

````markdown
```python exec="true" session="hess-fmt" source="above"
import jax
import jax.numpy as jnp
from asdex import hessian

def g(x):
    return jnp.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2)

x = jnp.arange(1.0, 6.0)

H_dense = jax.jit(hessian(g, x, output_format="dense"))(x)  # jax.Array
H_csr = hessian(g, x, output_format="scipy_csr")(x)         # scipy.sparse.csr_array
```

```python exec="true" session="hess-fmt"
print(f"""```
dense:     {type(H_dense).__name__} {H_dense.shape}
scipy_csr: {type(H_csr).__name__} nnz={H_csr.nnz}
```""")
```
````

### Section 7: Reducing Peak Memory with Chunking (#149)

**Jacobian guide** (`## Reducing Peak Memory with Chunking`):

````markdown
## Reducing Peak Memory with Chunking

Each color requires one VJP/JVP, and by default `asdex` evaluates **all** colors in a single `jax.vmap` batch.
For large patterns with many colors on memory-constrained hardware, `chunk_size` caps how many colors run in parallel:
chunks are processed sequentially via `jax.lax.map`, lowering the peak memory usage:

```python exec="true" session="jac-chunk" source="above"
import jax
import jax.numpy as jnp
from asdex import jacobian

def f(x):
    return (x[1:] - x[:-1]) ** 2

x = jnp.arange(1.0, 101.0)

# Evaluate at most 16 colors in parallel at a time:
J = jax.jit(jacobian(f, x, chunk_size=16))(x)
```

```python exec="true" session="jac-chunk"
print(f"```\nJ: {type(J).__name__} {J.shape}, nse={J.nse}\n```")
```

The result is identical to the default (`chunk_size=None`), only peak memory and runtime change.
`chunk_size` is accepted by [`jacobian`](../reference/index.md#asdex.jacobian),
[`value_and_jacobian`](../reference/index.md#asdex.value_and_jacobian), and
[`jacobian_from_coloring`](../reference/index.md#asdex.jacobian_from_coloring).

**Hessian guide** (`## Reducing Peak Memory with Chunking`): identical prose with a
`hessian` call (`chunk_size` caps HVPs per batch):

````markdown
```python exec="true" session="hess-chunk" source="above"
import jax
import jax.numpy as jnp
from asdex import hessian

def g(x):
    return jnp.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2)

x = jnp.arange(1.0, 101.0)

H = jax.jit(hessian(g, x, chunk_size=16))(x)  # at most 16 HVPs in parallel
```

```python exec="true" session="hess-chunk"
print(f"```\nH: {type(H).__name__} {H.shape}, nse={H.nse}\n```")
```
````

### New page: `docs/how-to/verification.md` (issue-#152 comment)

Both guides today end with a near-identical *Verifying Results* section. Lift that content
into one task-oriented page covering both Jacobians and Hessians, and replace each guide's
section with a short pointer. Verified body:

````markdown
# Verifying Correctness

asdex's [sparsity patterns](../explanation/global-sparsity.md) should always be conservative,
but a bug in [sparsity detection](../explanation/sparsity-detection.md) could drop a nonzero,
resulting in wrong Jacobians or Hessians.
Verify asdex' results against vanilla JAX at least once on every new function.
This guide shows you how.

## Jacobians

[`check_jacobian_correctness`][asdex.check_jacobian_correctness] compares asdex's sparse Jacobian against a JAX reference.
It returns silently on success and raises a [`VerificationError`][asdex.VerificationError] on mismatch.

```python exec="true" session="verify" source="above"
import jax.numpy as jnp
from asdex import jacobian_coloring, check_jacobian_correctness

def f(x):
    return (x[1:] - x[:-1]) ** 2

x = jnp.arange(1.0, 11.0)
coloring = jacobian_coloring(f, x)
check_jacobian_correctness(f, x, coloring)  # silent ⇒ correct
```

```python exec="true" session="verify"
print("```\nJacobian verified ✓\n```")
```

By default this uses `method="matvec"`, computing randomized matrix-vector products (i.e., JVPs, VJPs, or HVPs, depending on the coloring).
This is cheap, O(k) in the number of probes, and scalable.
You can tune the probes, tolerances, and seed:

```python exec="true" session="verify" source="above"
check_jacobian_correctness(f, x, coloring, num_probes=50, rtol=1e-5, atol=1e-5, seed=42)
```

For an exact but expensive element-wise comparison against the full dense Jacobian,
use `method="dense"`:

```python exec="true" session="verify" source="above"
check_jacobian_correctness(f, x, coloring, method="dense")
```

!!! warning "Dense comparison is expensive"

    `method="dense"` materializes the full dense Jacobian (O(n²)),
    so reserve it for small problems.

## Hessians

[`check_hessian_correctness`][asdex.check_hessian_correctness] mirrors the Jacobian API:

```python exec="true" session="verify" source="above"
from asdex import hessian_coloring, check_hessian_correctness

def g(x):
    return jnp.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2)

coloring = hessian_coloring(g, x)
check_hessian_correctness(g, x, coloring)              # matvec (default)
check_hessian_correctness(g, x, coloring, method="dense")  # exact, expensive
```

```python exec="true" session="verify"
print("```\nHessian verified ✓\n```")
```

## Validating a coloring directly

To check a coloring without evaluating derivatives,
use the coloring validators,
which raise [`InvalidColoringError`][asdex.InvalidColoringError] on a bad assignment:
[`check_coloring_rows`][asdex.check_coloring_rows] (reverse mode),
[`check_coloring_cols`][asdex.check_coloring_cols] (forward mode), and
[`check_coloring_symmetric`][asdex.check_coloring_symmetric] (Hessians).
````

### Inbound links to `verification.md`

The new page is the single home for correctness checking,
so every existing place in the docs that mentions verification points at it.
Four pages link in; each retarget is given below with its exact before/after.

**1. Both how-to guides, slot 13** (replaces the lifted *Verifying Results* section):

```markdown
## Verifying Results

Always check a new function against vanilla JAX at least once.
See [Verifying Correctness](verification.md) for
[`check_jacobian_correctness`][asdex.check_jacobian_correctness] / [`check_hessian_correctness`][asdex.check_hessian_correctness],
the `matvec` vs `dense` methods, and tolerance options.
```

**2. Both how-to guides, top-of-page tip** (the body moves to the new page,
so the tip's final line retargets from the in-page anchor to it).
Before:

```markdown
!!! tip "Verify correctness at least once"

    asdex's [sparsity patterns](../explanation/global-sparsity.md) should always be conservative,
    but a bug in [sparsity detection](../explanation/sparsity-detection.md) could cause missing nonzeros.
    Always verify against vanilla JAX at least once on a new function.
    See [Verifying Results](#verifying-results) below.
```

After:

```markdown
!!! tip "Verify correctness at least once"

    asdex's [sparsity patterns](../explanation/global-sparsity.md) should always be conservative,
    but a bug in [sparsity detection](../explanation/sparsity-detection.md) could cause missing nonzeros,
    resulting in wrong Jacobians or Hessians.
    Always verify against vanilla JAX at least once on a new function.
    See [Verifying Correctness](verification.md).
```

**3. `tutorials/getting-started.md`**: the closing line of the verification paragraph
currently deep-links into the Jacobian guide's now-removed *Verifying Results* anchor.
Retarget it to the new page.
Before:

```markdown
`asdex` also provides [`check_jacobian_correctness`][asdex.check_jacobian_correctness]
as a convenience for this comparison —
see [Verifying Results](../how-to/jacobians.md#verifying-results).
```

After:

```markdown
`asdex` also provides [`check_jacobian_correctness`][asdex.check_jacobian_correctness]
as a convenience for this comparison, see [Verifying Correctness](../how-to/verification.md).
```

**4. `explanation/global-sparsity.md`**: the *"correctness comes first"* paragraph
explains why patterns are conservative but offers no way to act on it.
Add a one-line pointer so the reader can check conservatism in practice.
Before:

```markdown
This asymmetry is why `asdex` errs on the side of conservatism:
correctness comes first.
```

After:

```markdown
This asymmetry is why `asdex` errs on the side of conservatism:
correctness comes first.
You can confirm a detected pattern against vanilla JAX.
See [Verifying Correctness](../how-to/verification.md).
```

After these edits, a repo-wide search for `#verifying-results` returns no hits,
which the strict build double-checks (a stale anchor would fail the build).

---

## Edits to `README.md`: the deliverable of #152

Insert a **Features** section between "Example" and "Documentation". Each feature links to
its section in the **Jacobian** guide (absolute URLs, since the README also renders on
GitHub/PyPI). Anchors are the slugified section headers.

```markdown
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
```

Anchor sanity-check (mkdocs/Material slugify lowercases, spaces → `-`, drops other
punctuation): `#getting-the-primal-value-too`, `#multiple-inputs-and-outputs`,
`#pytree-inputs-and-outputs`, `#auxiliary-outputs`, `#output-formats`,
`#reducing-peak-memory-with-chunking`, `#precomputing-the-colored-pattern`,
`#manually-providing-a-sparsity-pattern`. (Verification links to its own page, not an
anchor.) The strict build (below) will flag any anchor that does not resolve.

## Edits to `docs/index.md`: mirror the README

The docs **Home** page (`docs/index.md`, wired as `nav: Home → Overview`)
tracks `README.md` as closely as the two platforms allow,
including the new **Features** list.
The two files are already structurally parallel
(intro → Installation → Example → link list → Acknowledgements → Citation);
this adds the missing **Features** section.

Add the same **Features** block defined above for the README,
in the same slot (between the example and the link list),
swapping the absolute `https://adrianhill.de/asdex/...` URLs for the relative `.md` links
the rest of `index.md` already uses (e.g. `explanation/sparsity-detection.md`),
so the strict build validates every target and anchor.
Keep the two lists in lock-step: a future edit to one is mirrored in the other.

The remaining divergences are intentional and platform-forced, so do not reconcile them:

- the README's centered HTML logo/title header vs. `index.md`'s `# asdex` heading
  (the Material theme already renders the logo in the site header),
- absolute URLs in the README (it also renders on GitHub/PyPI, where relative `.md` links
  would not resolve) vs. relative `.md` links in `index.md`,
- the runnable example body (the README shows the braille preview;
  `index.md` keeps its shorter `jax.jit(asdex.jacobian(f, x))` snippet).

## `docs/mkdocs.yml`

One new nav entry for the verification page, under **How-To Guides** after *Sparse
Hessians*:

```yaml
  - How-To Guides:
      - Sparse Jacobians: how-to/jacobians.md
      - Sparse Hessians: how-to/hessians.md
      - Verifying Correctness: how-to/verification.md   # NEW
      - Visualization: how-to/visualization.md
      - "Example: Brusselator PDE": how-to/brusselator.md
```

---

## `docs/CLAUDE.md`

`docs/CLAUDE.md` is the guide future agents read before touching the docs,
so it must describe the docs *as they end up*, not as they were.
**As the final step of this work, update it to reflect the updated `/docs` folder.**

This is a general housekeeping pass:
re-read `docs/CLAUDE.md` end-to-end after the doc edits
and fix anything it now states inaccurately, not only the points listed above.

---

## Out of scope: public compressed API (#153)

Subissue [#153](https://github.com/adrhill/asdex/issues/153)
(*feat: public API for compressed Jacobians/Hessians*) is **explicitly excluded** from this plan.
It is a feature, not documentation, and is tracked separately.

## Execution order

A single PR closes all docs issues,
split into two commits so the diff is easy to review:
a pure restructuring with no content changes,
then the additive content.

**Commit 1 — restructure, no content changes.**
Reorganize both guides into the `## Basics` / `## Advanced` two-tier skeleton above:
add the two tier headings,
demote every existing section from `##` to `###` under the right tier,
and move *Choosing Row vs Column Coloring* / *Symmetric Coloring* + *Choosing an HVP Mode*
and *Separate Detection and Coloring* to the top of *Advanced*.
No prose, heading text, code, or anchors change;
this commit only moves and re-levels existing sections,
so a reviewer can confirm at a glance that nothing was rewritten.
Conventional commit: `docs: restructure how-to guides into Basics and Advanced sections`.

**Commit 2 — add the new content.**
Append the five **NEW** feature sections
(*Multiple Inputs and Outputs*, *PyTree Inputs and Outputs*, *Auxiliary Outputs*,
*Output Formats*, *Reducing Peak Memory with Chunking*)
as `###` at the end of each guide's *Advanced* section;
add the new `how-to/verification.md` page (+ nav entry)
and trim each guide's *Verifying Results* to a pointer;
retarget every inbound verification link (the four pages above);
drop the now-redundant multi-dimensional line in the Jacobian *Basic Usage*;
add the **Features** list to both `README.md` and `docs/index.md`;
and finally update `docs/CLAUDE.md` to match the new structure.
Conventional commit: `docs: document output formats, pytrees, aux, chunking, and verification`.

Both commits land in one PR.
Closes #148, #149, #150, #151, #152.

---

## Previewing the docs

All snippets are executed by `markdown-exec` at build time, so serving the site renders
their real output, the fastest way to visually confirm the changes.

Start by running the strict build once (the CI gate); it executes every snippet
and fails on any broken link, bad anchor, or warning:

```bash
uv run mkdocs build --strict -f docs/mkdocs.yml
```

If this fails, fix the docs.
Afterwards, start a live-reloading preview from the repo root:

```bash
uv run mkdocs serve -f docs/mkdocs.yml
```

Then open <http://127.0.0.1:8000> in the browser for the maintainers, such that they can confirm the correctness of the build.

!!! note "Restart to refresh"

    Per `docs/CLAUDE.md`, live reload is unreliable for `exec="true"` blocks, so stop
    (`Ctrl-C`) and restart `mkdocs serve` to see edits.
