# Computing Sparse Hessians

`asdex` computes sparse Hessians for scalar-valued functions
\(f: \mathbb{R}^n \to \mathbb{R}\)
using symmetric coloring and forward-over-reverse AD.

!!! tip "Verify correctness at least once"

    asdex's [sparsity patterns](../explanation/global-sparsity.md) should always be conservative,
    but a bug in [sparsity detection](../explanation/sparsity-detection.md) could cause missing nonzeros,
    resulting in wrong Jacobians or Hessians.
    Always verify against vanilla JAX at least once on a new function.
    See [Verifying Correctness](verification.md).

## Basics

### Basic Usage

Pass your scalar-valued function and a sample input to [`hessian`](../reference/index.md#asdex.hessian):

```python exec="true" session="hess-basic" source="above"
import jax
import jax.numpy as jnp
from asdex import hessian

def g(x):
    return jnp.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2)

x = jnp.zeros(100)  # any input with the correct shape and dtype works for detection
hess_fn = jax.jit(hessian(g, x))

H = hess_fn(x)
```

This runs the computationally expensive sparsity detection and coloring steps when defining `hess_fn`.
Subsequent calls to `hess_fn` only need to perform the cheap decompression step.
The result is a JAX [BCOO](https://docs.jax.dev/en/latest/jax.experimental.sparse.html) sparse matrix.

The same function can be reused across evaluations at different inputs:

```python
for x in inputs:
    H = hess_fn(x)
```

For a vector input,
the Hessian is the familiar 2D matrix of shape \((n, n)\),
where \(n\) is the number of input elements.

### Getting the Primal Value Too

Use [`value_and_hessian`](../reference/index.md#asdex.value_and_hessian)
or [`value_and_hessian_from_coloring`](../reference/index.md#asdex.value_and_hessian_from_coloring)
to get `(f(x), H)` without a redundant forward pass.

```python exec="true" session="hess-value" source="above"
import jax.numpy as jnp
from asdex import value_and_hessian

def g(x):
    return jnp.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2)

x = jnp.arange(1.0, 6.0)
y, H = value_and_hessian(g, x)(x)  # y is the primal g(x), H is the sparse Hessian
```

### Precomputing the Colored Pattern

For more control,
precompute the coloring explicitly:

```python exec="true" session="hess-precompute" source="above"
import jax.numpy as jnp
from asdex import hessian_coloring, hessian_from_coloring

def g(x):
    return jnp.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2)

x = jnp.zeros(100)
coloring = hessian_coloring(g, x)
```

```python exec="true" session="hess-precompute"
print(f"```\n{coloring}\n```")
```

This is useful when you want to visually inspect the coloring for correctness,
or save it to disk to avoid recomputation.
Pass the coloring to [`hessian_from_coloring`](../reference/index.md#asdex.hessian_from_coloring) to compute the Hessian:

```python
hess_fn = jax.jit(hessian_from_coloring(g, coloring))

for x in inputs:
    H = hess_fn(x)
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
from asdex import hessian_coloring

x = jnp.zeros(100)
coloring = hessian_coloring(g, x)
coloring.save("colored.npz")
```

```python
from asdex import ColoredPattern, hessian_from_coloring

coloring = ColoredPattern.load("colored.npz")
hess_fn = jax.jit(hessian_from_coloring(g, coloring))
```

[`SparsityPattern`](../reference/index.md#asdex.SparsityPattern) supports the same `save`/`load` interface.

### Manually Providing a Sparsity Pattern

You can provide a sparsity pattern manually if you already know it ahead of time.
Create a `SparsityPattern` from coordinate arrays, a dense matrix, or a JAX BCOO matrix.

From a dense boolean or numeric matrix:

```python exec="true" session="hess" source="above"
import numpy as np
from asdex import SparsityPattern

dense = np.array([[1, 1, 0, 0],
                  [1, 1, 1, 0],
                  [0, 1, 1, 1],
                  [0, 0, 1, 1]])
sparsity = SparsityPattern.from_dense(dense)
```
```python exec="true" session="hess"
print(f"```\n{sparsity}\n```")
```

From row and column index arrays:

```python exec="true" session="hess" source="above"
sparsity = SparsityPattern.from_coo(
    rows=[0, 0, 1, 1, 1, 2, 2, 2, 3, 3],
    cols=[0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
    shape=(4, 4),
)
```
```python exec="true" session="hess"
print(f"```\n{sparsity}\n```")
```

From a JAX BCOO sparse matrix:
```python
sparsity = SparsityPattern.from_bcoo(bcoo_matrix)
```

Finally, color the sparsity pattern and compute the Hessian:
```python
from asdex import hessian_coloring_from_sparsity, hessian_from_coloring

coloring = hessian_coloring_from_sparsity(sparsity)
hess_fn = jax.jit(hessian_from_coloring(f, coloring))
H = hess_fn(x)
```

### Separate Detection and Coloring

For even more control, you can split detection and coloring:

```python
import jax.numpy as jnp
from asdex import hessian_sparsity, hessian_coloring_from_sparsity

x = jnp.zeros(100)
sparsity = hessian_sparsity(g, x)
coloring = hessian_coloring_from_sparsity(sparsity)
```

Since the Hessian is the Jacobian of the gradient,
`hessian_sparsity` simply calls `jacobian_sparsity(jax.grad(f), x)`.
The [sparsity interpreter](../explanation/sparsity-detection.md) composes naturally with JAX's autodiff transforms.

This is useful when you want to manually provide a sparsity pattern.

### Verifying Results

Always check a new function against vanilla JAX at least once.
See [Verifying Correctness](verification.md) for
[`check_jacobian_correctness`][asdex.check_jacobian_correctness] / [`check_hessian_correctness`][asdex.check_hessian_correctness],
the `matvec` vs `dense` methods, and tolerance options.

## Advanced

### Symmetric Coloring

Hessians are symmetric (\(H = H^\top\)),
and `asdex` exploits this with *star coloring*
(Gebremedhin et al., 2005).
Symmetric coloring typically needs fewer colors than row or column coloring,
since both \(H_{ij}\) and \(H_{ji}\) can be recovered from a single coloring.

The convenience functions [`hessian_coloring`](../reference/index.md#asdex.hessian_coloring) and [`hessian`](../reference/index.md#asdex.hessian) use symmetric coloring automatically.
Here we use the [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function),
a classic optimization benchmark whose Hessian is tridiagonal:

\[f(x) = \sum_{i=1}^{n-1} \left[(1 - x_i)^2 + 100\,(x_{i+1} - x_i^2)^2\right]\]

```python exec="true" session="hess" source="above"
import jax.numpy as jnp
from asdex import hessian_coloring

def rosenbrock(x):
    return jnp.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2)

x = jnp.zeros(100)
coloring = hessian_coloring(rosenbrock, x)
```

```python exec="true" session="hess"
print(f"```\n{coloring}\n```")
```

### Choosing an HVP Mode

By default, `hessian` uses forward-over-reverse AD to compute Hessian-vector products.
You can select a different AD composition strategy via the `mode` parameter:

```python
import jax.numpy as jnp
from asdex import hessian

x = jnp.zeros(100)
hess_fn_for = jax.jit(hessian(f, x, mode="fwd_over_rev"))  # default
hess_fn_rof = jax.jit(hessian(f, x, mode="rev_over_fwd"))
hess_fn_ror = jax.jit(hessian(f, x, mode="rev_over_rev"))
```

All three modes produce the same mathematical result.
They differ in their performance and memory trade-offs:

- **`fwd_over_rev`** (default): generally the fastest under JIT.
- **`rev_over_fwd`**: can use less memory than forward-over-reverse for functions with many intermediates.
- **`rev_over_rev`**: avoids forward-mode entirely, which is useful when forward-mode is expensive or unsupported.

!!! tip

    When in doubt, stick with the default `"fwd_over_rev"`.
    It is the most widely used and typically the most efficient under `jax.jit`.

### Multiple Inputs

`asdex` mirrors `jax.hessian`:
it differentiates functions of several arguments,
selecting which arguments to differentiate with `argnums`.
A Hessian requires a scalar output, so there is no multiple-output case.

Pass a sample value for each positional argument,
and select the ones to differentiate with `argnums`.
With a tuple `argnums` the result is a nested `(input_tree, input_tree)` grid,
mirroring `jax.hessian`:
`H[i][j]` is the second derivative with respect to argument `i` and argument `j`,
so the full block grid is shown rather than a single corner.

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
print(f"""```
len(H):        {len(H)}
len(H[0]):     {len(H[0])}
H[0][0].shape: {H[0][0].shape}
H[0][1].shape: {H[0][1].shape}
H[1][1].shape: {H[1][1].shape}
```""")
```

### PyTree Inputs

A single argument can itself be an arbitrary [PyTree](https://docs.jax.dev/en/latest/pytrees.html),
such as a dictionary of parameters.
For a PyTree argument the Hessian is a matching nested structure of blocks,
here a dict-of-dicts, where `H[i][j]` couples leaves `i` and `j`:

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
print(f"""```
keys:              {sorted(H)}
H['a']['a'].shape: {H['a']['a'].shape}
H['a']['b'].shape: {H['a']['b'].shape}
```""")
```

### Auxiliary Outputs

Set `has_aux=True` when your function returns `(output, auxiliary_data)`, mirroring `jax.hessian`.
The auxiliary data is passed through untouched, useful for diagnostics, intermediate values, or model state.

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
H.shape:     {H.shape}
aux['norm']: {float(aux['norm']):.3f}
```""")
```

[`value_and_hessian`](../reference/index.md#asdex.value_and_hessian) nests aux next to the value,
matching `jax.value_and_grad` ordering, giving `((value, aux), H)`:

```python exec="true" session="hess-aux" source="above"
from asdex import value_and_hessian

(value, aux), H = value_and_hessian(g, x, has_aux=True)(x)
```

```python exec="true" session="hess-aux"
print(f"""```
value.shape: {value.shape}
aux['norm']: {float(aux['norm']):.3f}
```""")
```

The auxiliary data may hold arbitrary Python objects, not just JAX arrays.
It is extracted from the forward pass that AD already runs,
so returning it adds no extra evaluation of `f`.

### Output Formats

By default, `asdex` returns sparse matrices as JAX [BCOO](https://docs.jax.dev/en/latest/jax.experimental.sparse.html) arrays.
The `output_format` argument selects a different container.
It is accepted by [`hessian`](../reference/index.md#asdex.hessian),
its `value_and_*` variant, and the `*_from_coloring` variants.

| `output_format` | Returned type | JIT-able by caller |
|-----------------|---------------|--------------------|
| `"bcoo"` (default) | `jax.experimental.sparse.BCOO` | yes |
| `"dense"` | `jax.Array` | yes |
| `"numpy_dense"` | `numpy.ndarray` | no |
| `"scipy_coo"` | `scipy.sparse.coo_array` | no |
| `"scipy_csr"` | `scipy.sparse.csr_array` | no |
| `"scipy_csc"` | `scipy.sparse.csc_array` | no |

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
H_dense:  {type(H_dense).__name__}  shape={H_dense.shape}
H_csr:    {type(H_csr).__name__}  nnz={H_csr.nnz}
```""")
```

!!! warning "Host formats are not JIT-able by the caller"

    `"numpy_dense"` and the scipy formats produce non-JAX arrays,
    so you cannot wrap the returned function in `jax.jit`.
    `asdex` JIT-compiles their core internally, so they stay fast anyway.
    Just call them directly:

    ```python
    H = hessian(g, x, output_format="numpy_dense")(x)  # do NOT jax.jit this
    ```

!!! info "SciPy formats are 2D-only"

    SciPy sparse arrays are strictly 2D.
    They require the input to be a single flat 1D array.
    `asdex` flattens and checks the full input structure up front.
    Any other shape, such as a multi-dimensional array, multiple arguments,
    or an arbitrarily nested PyTree, raises a clear `ValueError` rather than a wrong result.
    Note that SciPy is an optional dependency. Install it via `pip install 'asdex[scipy]'`.

Structural non-zeros that happen to be numerically zero at the evaluation point are kept as explicit entries in the `BCOO` and scipy outputs,
so the structure always matches the detected [global sparsity pattern](../explanation/global-sparsity.md) and is independent of `x`.

### Reducing Peak Memory with Chunking

Each color requires one HVP, and by default `asdex` evaluates **all** colors in a single `jax.vmap` batch.
For large patterns with many colors on memory-constrained hardware, `chunk_size` caps how many colors run in parallel:
chunks are processed sequentially via `jax.lax.map`, lowering the peak memory usage:

```python exec="true" session="hess-chunk" source="above"
import jax
import jax.numpy as jnp
from asdex import hessian

def g(x):
    return jnp.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2)

x = jnp.arange(1.0, 101.0)

H = jax.jit(hessian(g, x, chunk_size=16))(x)  # at most 16 HVPs in parallel
```

The result is identical to the default (`chunk_size=None`), only peak memory and runtime change.
`chunk_size` is accepted by [`hessian`](../reference/index.md#asdex.hessian),
[`value_and_hessian`](../reference/index.md#asdex.value_and_hessian), and
[`hessian_from_coloring`](../reference/index.md#asdex.hessian_from_coloring).

### Skipping Decompression

To access the raw compressed matrix \(B\) rather than the assembled sparse Hessian
(to feed a custom solver, cross-check against a reference, or decompress lazily),
use [`compressed_hessian`](../reference/index.md#asdex.compressed_hessian) and
[`compressed_hessian_from_coloring`](../reference/index.md#asdex.compressed_hessian_from_coloring).
They run the same detect-and-color steps as [`hessian`](../reference/index.md#asdex.hessian),
but stop at \(B\), the dense matrix of one HVP per color of shape \((\text{num\_colors}, n)\),
where \(n\) is the input size.

```python exec="true" session="hess-compress" source="above"
import jax
import jax.numpy as jnp
from asdex import compressed_hessian_from_coloring, decompress, decompress_data, hessian_coloring

def g(x):
    return jnp.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2)

x = jnp.arange(1.0, 6.0)
coloring = hessian_coloring(g, x)

compressed_fn = jax.jit(compressed_hessian_from_coloring(g, coloring))
B = compressed_fn(x)  # the dense compressed matrix, shape (num_colors, n)
```

`B` is a plain `jax.Array`, so the returned function stays jit-able by the caller.
The compressed functions take no `output_format`, since formatting is the job of decompression.
Each one has a `value_and_*` variant
([`value_and_compressed_hessian`](../reference/index.md#asdex.value_and_compressed_hessian)
and its `*_from_coloring` form) that also returns the primal value.

[`decompress`](../reference/index.md#asdex.decompress) turns \(B\) back into the sparse matrix
in any [output format](#output-formats).
Unlike [`hessian`](../reference/index.md#asdex.hessian),
it always returns the flat 2-D \((n, n)\) matrix, regardless of input PyTree structure:

```python exec="true" session="hess-compress" source="above"
H_bcoo = decompress(coloring, B)                          # BCOO (default)
H_dense = decompress(coloring, B, output_format="dense")  # jax.Array
```

For full control, [`decompress_data`](../reference/index.md#asdex.decompress_data) is the jittable
primitive underneath `decompress`.
It returns just the structural non-zero values as a `jax.Array` of shape \((\text{nnz},)\) in pattern order,
ready to pair with `coloring.sparsity.rows` and `coloring.sparsity.cols` to build a custom container:

```python exec="true" session="hess-compress" source="above"
data = decompress_data(coloring, B)  # the nnz values, in pattern order
rows = coloring.sparsity.rows        # row index of each value
cols = coloring.sparsity.cols        # column index of each value
# data, rows, and cols share one pattern order,
# so data[k] is the Hessian entry at (rows[k], cols[k]).
```

```python exec="true" session="hess-compress"
print(f"""```
B.shape:       {B.shape}
type(H_bcoo):  {type(H_bcoo).__name__}  shape={H_bcoo.shape}
data.shape:    {data.shape}
data[0]:       {float(data[0]):.1f}  at (rows[0], cols[0]) = ({int(rows[0])}, {int(cols[0])})
```""")
```

Because `decompress_data` always returns a `jax.Array`,
it composes inside `jax.jit` and can feed a custom solver,
whereas the host formats from `decompress` (`"numpy_dense"` and the scipy formats) cannot.
