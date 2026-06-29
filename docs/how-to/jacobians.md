# Computing Sparse Jacobians

`asdex` computes sparse Jacobians for functions
\(f: \mathbb{R}^n \to \mathbb{R}^m\)
using [row or column coloring](../explanation/coloring.md) with forward- or reverse-mode AD.

!!! tip "Verify correctness at least once"

    asdex's [sparsity patterns](../explanation/global-sparsity.md) should always be conservative,
    but a bug in [sparsity detection](../explanation/sparsity-detection.md) could cause missing nonzeros,
    resulting in wrong Jacobians or Hessians.
    Always verify against vanilla JAX at least once on a new function.
    See [Verifying Correctness](verification.md).

## Basics

### Basic Usage

Pass your function and a sample input to [`jacobian`](../reference/index.md#asdex.jacobian):

```python exec="true" session="jac-basic" source="above"
import jax
import jax.numpy as jnp
from asdex import jacobian

def f(x):
    return (x[1:] - x[:-1]) ** 2

x = jnp.zeros(1000)  # any input with the correct shape and dtype works for detection
jac_fn = jax.jit(jacobian(f, x))

J = jac_fn(x)
```

This runs the computationally expensive sparsity detection and coloring steps when defining `jac_fn`.
Subsequent calls to `jac_fn` only need to perform the cheap decompression step.
The result is a JAX [BCOO](https://docs.jax.dev/en/latest/jax.experimental.sparse.html) sparse matrix.

The same function can be reused across evaluations at different inputs:

```python
for x in inputs:
    J = jac_fn(x)
```

For a vector input and vector output,
the Jacobian is the familiar 2D matrix of shape \((m, n)\),
with \(m\) the number of output elements and \(n\) the number of input elements.

### Getting the Primal Value Too

Use [`value_and_jacobian`](../reference/index.md#asdex.value_and_jacobian)
or [`value_and_jacobian_from_coloring`](../reference/index.md#asdex.value_and_jacobian_from_coloring)
to get `(f(x), J)` without a redundant forward pass.

```python exec="true" session="jac-value" source="above"
import jax.numpy as jnp
from asdex import value_and_jacobian

def f(x):
    return (x[1:] - x[:-1]) ** 2

x = jnp.arange(1.0, 6.0)
y, J = value_and_jacobian(f, x)(x)  # y is the primal f(x), J is the sparse Jacobian
```

### Precomputing the Colored Pattern

For more control,
precompute the coloring explicitly:

```python exec="true" session="jac-precompute" source="above"
import jax.numpy as jnp
from asdex import jacobian_coloring, jacobian_from_coloring

def f(x):
    return (x[1:] - x[:-1]) ** 2

x = jnp.zeros(100)
coloring = jacobian_coloring(f, x)
```

```python exec="true" session="jac-precompute"
print(f"```\n{coloring}\n```")
```

This is useful when you want to visually inspect the coloring for correctness,
or save it to disk to avoid recomputation.
Pass the coloring to [`jacobian_from_coloring`](../reference/index.md#asdex.jacobian_from_coloring) to compute the Jacobian:

```python
jac_fn = jax.jit(jacobian_from_coloring(f, coloring))

for x in inputs:
    J = jac_fn(x)
```

!!! tip

    If your `coloring` looks wrong or overly dense,
    please help out `asdex`'s development by
    [reporting it](https://github.com/adrhill/asdex/issues).
    These reports directly drive improvements
    and are one of the most impactful ways to contribute.

### Saving and Loading Patterns

Save a coloring to disk and reload it in a later session:

```python
import jax.numpy as jnp
from asdex import jacobian_coloring

x = jnp.zeros(1000)
coloring = jacobian_coloring(f, x)
coloring.save("colored.npz")
```

```python
from asdex import ColoredPattern, jacobian_from_coloring

coloring = ColoredPattern.load("colored.npz")
jac_fn = jax.jit(jacobian_from_coloring(f, coloring))
```

[`SparsityPattern`](../reference/index.md#asdex.SparsityPattern) supports the same `save`/`load` interface.

### Manually Providing a Sparsity Pattern

You can provide a sparsity pattern manually if you already know it ahead of time.
Create a `SparsityPattern` from coordinate arrays, a dense matrix, or a JAX BCOO matrix.

From a dense boolean or numeric matrix:

```python exec="true" session="jac" source="above"
import numpy as np
from asdex import SparsityPattern

dense = np.array([[1, 1, 0, 0],
                  [0, 1, 1, 0],
                  [0, 0, 1, 1]])
sparsity = SparsityPattern.from_dense(dense)
```
```python exec="true" session="jac"
print(f"```\n{sparsity}\n```")
```

From row and column index arrays:

```python exec="true" session="jac" source="above"
sparsity = SparsityPattern.from_coo(
    rows=[0, 0, 1, 1, 2, 2],
    cols=[0, 1, 1, 2, 2, 3],
    shape=(3, 4),
)
```
```python exec="true" session="jac"
print(f"```\n{sparsity}\n```")
```

From a JAX BCOO sparse matrix:
```python
sparsity = SparsityPattern.from_bcoo(bcoo_matrix)
```

Finally, color the sparsity pattern and compute the Jacobian:
```python
from asdex import jacobian_coloring_from_sparsity, jacobian_from_coloring

coloring = jacobian_coloring_from_sparsity(sparsity)
jac_fn = jax.jit(jacobian_from_coloring(f, coloring))
J = jac_fn(x)
```

### Separate Detection and Coloring

For even more control, you can split detection and coloring:

```python
import jax.numpy as jnp
from asdex import jacobian_sparsity, jacobian_coloring_from_sparsity

x = jnp.zeros(1000)
sparsity = jacobian_sparsity(f, x)
coloring = jacobian_coloring_from_sparsity(sparsity, mode="fwd")
```

This is useful when you want to manually provide a sparsity pattern.

### Verifying Results

Always check a new function against vanilla JAX at least once.
See [Verifying Correctness](verification.md) for
[`check_jacobian_correctness`][asdex.check_jacobian_correctness] / [`check_hessian_correctness`][asdex.check_hessian_correctness],
the `matvec` vs `dense` methods, and tolerance options.

## Advanced

### Choosing Row vs Column Coloring

By default, `asdex` tries both row and column coloring
and picks whichever needs fewer colors:

```python exec="true" session="jac" source="above"
import jax.numpy as jnp
from asdex import jacobian_coloring

def f(x):
    return (x[1:] - x[:-1]) ** 2

x = jnp.zeros(100)

# Automatic selection (default):
coloring = jacobian_coloring(f, x)
```

```python exec="true" session="jac"
print(f"```\n{coloring}\n```")
```

You can also force a specific AD mode.
``"fwd"`` colors columns (uses JVPs, forward-mode AD),
``"rev"`` colors rows (uses VJPs, reverse-mode AD):

```python exec="true" session="jac" source="above"
# Force forward mode (column coloring, uses JVPs):
coloring = jacobian_coloring(f, x, mode="fwd")

# Force reverse mode (row coloring, uses VJPs):
coloring = jacobian_coloring(f, x, mode="rev")
```

```python exec="true" session="jac"
print(f"```\n{coloring}\n```")
```

The one-call [`jacobian`](../reference/index.md#asdex.jacobian) API accepts the same `mode` parameter:

```python
jac_fn = jax.jit(jacobian(f, x, mode="rev"))
```

When the number of colors is equal,
`asdex` prefers column coloring since JVPs are generally cheaper to compute in JAX.

### Multiple Inputs and Outputs

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

With an integer `argnums` (the default `0`) a single block is returned, not a tuple.
Arguments not named by `argnums` are still passed at call time and held fixed,
yet they can still influence the result.
Here `scale` is not differentiated, but it scales every entry of the Jacobian:

```python exec="true" session="jac-multi" source="above"
def scaled(x, scale):
    return scale * x ** 2

J1 = jacobian(scaled, x, 1.0, argnums=0)(x, 1.0)  # [ 1.  4.  6.]
J5 = jacobian(scaled, x, 5.0, argnums=0)(x, 5.0)  # [10. 20. 30.]
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
print(f"""```
len(J):        {len(J)}
len(J[0]):     {len(J[0])}
J[0][0].shape: {J[0][0].shape}
```""")
```

### PyTree Inputs and Outputs

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
keys:              {sorted(J)}
J['weight'].shape: {J['weight'].shape}
J['bias'].shape:   {J['bias'].shape}
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
keys:               {sorted(J)}
J['squared'].shape: {J['squared'].shape}
J['total'].shape:   {J['total'].shape}
```""")
```

### Auxiliary Outputs

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
J.shape:        {J.shape}
aux['mean_sq']: {float(aux['mean_sq']):.3f}
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
value.shape:    {value.shape}
aux['mean_sq']: {float(aux['mean_sq']):.3f}
```""")
```

The auxiliary data may hold arbitrary Python objects, not just JAX arrays.
It is extracted from the forward pass that AD already runs,
so returning it adds no extra evaluation of `f`.

### Output Formats

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
J_bcoo:       {type(J_bcoo).__name__}
J_dense:      {type(J_dense).__name__}  shape={J_dense.shape}
J_csr:        {type(J_csr).__name__}  nnz={J_csr.nnz}
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

### Reducing Peak Memory with Chunking

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

The result is identical to the default (`chunk_size=None`), only peak memory and runtime change.
`chunk_size` is accepted by [`jacobian`](../reference/index.md#asdex.jacobian),
[`value_and_jacobian`](../reference/index.md#asdex.value_and_jacobian), and
[`jacobian_from_coloring`](../reference/index.md#asdex.jacobian_from_coloring).

### Skipping Decompression

Sometimes you want the raw compressed matrix \(B\) rather than the assembled sparse Jacobian,
to feed a custom solver, cross-check against a reference, or decompress lazily.
[`compressed_jacobian`](../reference/index.md#asdex.compressed_jacobian) and
[`compressed_jacobian_from_coloring`](../reference/index.md#asdex.compressed_jacobian_from_coloring)
run the same detect-and-color steps as [`jacobian`](../reference/index.md#asdex.jacobian),
but stop at \(B\), the dense matrix of one VJP or JVP per color of shape \((\text{num\_colors}, \text{dim})\).
Here `dim` is the input size \(n\) in `"rev"` mode and the output size \(m\) in `"fwd"` mode.

```python exec="true" session="jac-compress" source="above"
import jax
import jax.numpy as jnp
from asdex import compressed_jacobian_from_coloring, decompress, decompress_data, jacobian_coloring

def f(x):
    return (x[1:] - x[:-1]) ** 2

x = jnp.arange(1.0, 6.0)
coloring = jacobian_coloring(f, x)

compressed_fn = jax.jit(compressed_jacobian_from_coloring(f, coloring))
B = compressed_fn(x)  # the dense compressed matrix, shape (num_colors, dim)
```

`B` is a plain `jax.Array`, so the returned function stays jit-able by the caller.
The compressed functions take no `output_format`, since formatting is the job of decompression.
Each one has a `value_and_*` variant
([`value_and_compressed_jacobian`](../reference/index.md#asdex.value_and_compressed_jacobian)
and its `*_from_coloring` form) that also returns the primal value.

[`decompress`](../reference/index.md#asdex.decompress) turns \(B\) back into the sparse matrix
in any [output format](#output-formats).
Unlike [`jacobian`](../reference/index.md#asdex.jacobian),
it always returns the flat 2-D \((m, n)\) matrix, regardless of input or output PyTree structure:

```python exec="true" session="jac-compress" source="above"
J_bcoo = decompress(coloring, B)                          # BCOO (default)
J_dense = decompress(coloring, B, output_format="dense")  # jax.Array
```

For full control, [`decompress_data`](../reference/index.md#asdex.decompress_data) is the jittable
primitive underneath `decompress`.
It returns just the structural non-zero values as a `jax.Array` of shape \((\text{nnz},)\) in pattern order,
ready to pair with `coloring.sparsity.rows` and `coloring.sparsity.cols` to build a custom container:

```python exec="true" session="jac-compress" source="above"
data = decompress_data(coloring, B)  # the nnz values, in pattern order
rows = coloring.sparsity.rows        # row index of each value
cols = coloring.sparsity.cols        # column index of each value
# data, rows, and cols share one pattern order,
# so data[k] is the Jacobian entry at (rows[k], cols[k]).
```

```python exec="true" session="jac-compress"
print(f"""```
B.shape:     {B.shape}
J_bcoo:      {type(J_bcoo).__name__}  shape={J_bcoo.shape}
data.shape:  {data.shape}
first entry: J[{rows[0]}, {cols[0]}] = {float(data[0]):.1f}
```""")
```

Because `decompress_data` always returns a `jax.Array`,
it composes inside `jax.jit` and can feed a custom solver,
whereas the host formats from `decompress` (`"numpy_dense"` and the scipy formats) cannot.
