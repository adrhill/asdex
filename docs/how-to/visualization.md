# Visualizing Sparsity Patterns

## Printing Patterns

Every [`SparsityPattern`](../reference/index.md#asdex.SparsityPattern)
and [`ColoredPattern`](../reference/index.md#asdex.ColoredPattern)
has a built-in text representation that works without extra dependencies.

Small patterns use a dot display:

```python exec="true" session="vis" source="above"
import jax.numpy as jnp
from asdex import jacobian_sparsity

def f(x):
    return (x[1:] - x[:-1]) ** 2

x = jnp.zeros(10)
sparsity = jacobian_sparsity(f, x)
```

```python exec="true" session="vis"
print(f"```\n{sparsity}\n```")
```

Larger patterns automatically switch to a compact braille rendering:

```python exec="true" session="vis" source="above"
x = jnp.zeros(200)
sparsity = jacobian_sparsity(f, x)
```

```python exec="true" session="vis"
print(f"```\n{sparsity}\n```")
```

Printing a [`ColoredPattern`](../reference/index.md#asdex.ColoredPattern)
shows the original and compressed patterns side by side,
along with a summary of the coloring:

```python exec="true" session="vis" source="above"
from asdex import jacobian_coloring

coloring = jacobian_coloring(f, x)
```

```python exec="true" session="vis"
print(f"```\n{coloring}\n```")
```

## Matplotlib Plots

For matplotlib figures,
use [`asdex.spy`](../reference/index.md#asdex.spy).

!!! note "Optional dependency"

    Plotting requires matplotlib.
    Install it manually or using `pip install asdex[matplotlib]`.

### Sparsity Patterns

Pass a [`SparsityPattern`](../reference/index.md#asdex.SparsityPattern)
to [`asdex.spy`](../reference/index.md#asdex.spy):

```python
# mkdocs: render
import numpy as np
from asdex import SparsityPattern, spy

dense = np.array([
    [1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1],
])
sparsity = SparsityPattern.from_dense(dense)
spy(sparsity)
```

### Colored Patterns

Pass a [`ColoredPattern`](../reference/index.md#asdex.ColoredPattern) to `asdex.spy`
to color nonzeros by their color assignment:

```python
# mkdocs: render
import jax.numpy as jnp
from asdex import jacobian_coloring, spy

def f(x):
    return (x[1:] - x[:-1]) ** 2

x = jnp.zeros(20)
coloring = jacobian_coloring(f, x)
spy(coloring)
```

### Showing Compression

Set `compressed=True` to plot the compressed pattern after coloring.
Use subplots to show original and compressed patterns side by side:

```python
# mkdocs: render
import jax.numpy as jnp
import matplotlib.pyplot as plt
from asdex import jacobian_coloring, spy

def f(x):
    return (x[1:] - x[:-1]) ** 2

x = jnp.zeros(20)
coloring = jacobian_coloring(f, x)
m, n = coloring.sparsity.shape
c = coloring.num_colors

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
spy(coloring, ax=ax1)
spy(coloring, compressed=True, ax=ax2)
ax1.set_title(f"Sparse Jacobian ({m}×{n})")
ax2.set_title(f"Compressed Jacobian ({m}×{c})")
plt.tight_layout()
```

### Customizing Plots

`asdex.spy` accepts a `cmap` argument to change the color scheme,
as well as any keyword argument supported by `ax.imshow`:

```python
# mkdocs: render
import jax.numpy as jnp
import matplotlib.pyplot as plt
from asdex import jacobian_coloring, spy

def f(x):
    return (x[1:] - x[:-1]) ** 2

x = jnp.zeros(20)
coloring = jacobian_coloring(f, x)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
spy(coloring, ax=ax1)
spy(coloring, ax=ax2, cmap="viridis")
ax1.set_title("tab10 (default)")
ax2.set_title("viridis")
plt.tight_layout()
```

### Examples

Row coloring (reverse mode):

```python
# mkdocs: render
import jax.numpy as jnp
import matplotlib.pyplot as plt
from asdex import jacobian_coloring, spy

def f(x):
    return (x[1:] - x[:-1]) ** 2

x = jnp.zeros(20)
coloring = jacobian_coloring(f, x, mode="rev")
m, n = coloring.sparsity.shape
c = coloring.num_colors

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
spy(coloring, ax=ax1)
spy(coloring, compressed=True, ax=ax2)
ax1.set_title(f"Sparse Jacobian ({m}×{n})")
ax2.set_title(f"Compressed Jacobian ({c}×{n})")
plt.tight_layout()
```

Symmetric coloring (Hessian):

```python
# mkdocs: render
import jax.numpy as jnp
import matplotlib.pyplot as plt
from asdex import hessian_coloring, spy

def g(x):
    X = x.reshape(5, 5)
    return ((X[1:, :] - X[:-1, :]) ** 2).sum() + ((X[:, 1:] - X[:, :-1]) ** 2).sum()

x = jnp.zeros(25)
coloring = hessian_coloring(g, x)
n = coloring.sparsity.shape[0]
c = coloring.num_colors

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
spy(coloring, ax=ax1)
spy(coloring, compressed=True, ax=ax2)
ax1.set_title(f"Sparse Hessian ({n}×{n})")
ax2.set_title(f"Compressed Hessian ({n}×{c})")
plt.tight_layout()
```

Here is a larger example using the
[Brusselator PDE](brusselator.md) discretized on an \(8 \times 8\) grid,
giving a Jacobian of shape \(128 \times 128\):

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt
from asdex import jacobian_coloring, spy

N = 8
alpha = 10.0
dx = 1.0 / N

def brusselator_rhs(uv):
    u = uv[:N*N].reshape(N, N)
    v = uv[N*N:].reshape(N, N)
    def laplacian(w):
        return (
            jnp.roll(w, 1, axis=0) + jnp.roll(w, -1, axis=0)
            + jnp.roll(w, 1, axis=1) + jnp.roll(w, -1, axis=1)
            - 4 * w
        ) / dx**2
    du = 1.0 + u**2 * v - 4.4 * u + alpha * laplacian(u)
    dv = 3.4 * u - u**2 * v + alpha * laplacian(v)
    return jnp.concatenate([du.ravel(), dv.ravel()])

uv = jnp.zeros(2 * N * N)
coloring = jacobian_coloring(brusselator_rhs, uv)
m, n = coloring.sparsity.shape
c = coloring.num_colors

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
spy(coloring, ax=ax1)
spy(coloring, compressed=True, ax=ax2)
ax1.set_title(f"Sparse Jacobian ({m}×{n})")
ax2.set_title(f"Compressed Jacobian ({m}×{c})")
plt.tight_layout()
```

```python
# mkdocs: render
# mkdocs: hidecode
import jax.numpy as jnp
import matplotlib.pyplot as plt
from asdex import jacobian_coloring, spy

N = 8
alpha = 10.0
dx = 1.0 / N

def brusselator_rhs(uv, N=N, alpha=alpha, dx=dx):
    import jax.numpy as jnp
    u = uv[:N*N].reshape(N, N)
    v = uv[N*N:].reshape(N, N)
    def laplacian(w):
        return (
            jnp.roll(w, 1, axis=0) + jnp.roll(w, -1, axis=0)
            + jnp.roll(w, 1, axis=1) + jnp.roll(w, -1, axis=1)
            - 4 * w
        ) / dx**2
    du = 1.0 + u**2 * v - 4.4 * u + alpha * laplacian(u)
    dv = 3.4 * u - u**2 * v + alpha * laplacian(v)
    return jnp.concatenate([du.ravel(), dv.ravel()])

uv = jnp.zeros(2 * N * N)
coloring = jacobian_coloring(brusselator_rhs, uv)
m, n = coloring.sparsity.shape
c = coloring.num_colors

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
spy(coloring, ax=ax1)
spy(coloring, compressed=True, ax=ax2)
ax1.set_title(f"Sparse Jacobian ({m}×{n})")
ax2.set_title(f"Compressed Jacobian ({m}×{c})")
plt.tight_layout()
```
