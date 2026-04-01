# Visualizing Sparsity Patterns

## Printing Patterns

Every [`SparsityPattern`](../reference/index.md#asdex.SparsityPattern)
and [`ColoredPattern`](../reference/index.md#asdex.ColoredPattern)
has a built-in text representation that works anywhere — no extra dependencies needed.

Small patterns use a dot display:

```python exec="true" session="vis" source="above"
from asdex import jacobian_sparsity

def f(x):
    return (x[1:] - x[:-1]) ** 2

sparsity = jacobian_sparsity(f, input_shape=10)
```

```python exec="true" session="vis"
print(f"```\n{sparsity}\n```")
```

Larger patterns automatically switch to a compact braille rendering:

```python exec="true" session="vis" source="above"
sparsity = jacobian_sparsity(f, input_shape=200)
```

```python exec="true" session="vis"
print(f"```\n{sparsity}\n```")
```

Printing a [`ColoredPattern`](../reference/index.md#asdex.ColoredPattern)
shows the original and compressed patterns side by side,
along with a summary of the AD savings:

```python exec="true" session="vis" source="above"
from asdex import jacobian_coloring

coloring = jacobian_coloring(f, input_shape=200)
```

```python exec="true" session="vis"
print(f"```\n{coloring}\n```")
```

## Matplotlib Plots

For publication-quality figures,
use [`spy`](../reference/index.md#asdex.spy).

!!! note "Optional dependency"

    Plotting requires matplotlib.
    Install it with `pip install asdex[matplotlib]`.

### Sparsity Patterns

Pass a [`SparsityPattern`](../reference/index.md#asdex.SparsityPattern) to `spy`
to plot nonzeros as black markers:

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

Pass a [`ColoredPattern`](../reference/index.md#asdex.ColoredPattern) to `spy`
to color nonzeros by their color assignment:

```python
# mkdocs: render
from asdex import jacobian_coloring, spy

def f(x):
    return (x[1:] - x[:-1]) ** 2

coloring = jacobian_coloring(f, input_shape=20)
spy(coloring)
```

### Showing Compression

Set `compressed=True` to show the original and compressed patterns side by side.
This visualizes how graph coloring reduces the number of AD evaluations:

```python
# mkdocs: render
from asdex import jacobian_coloring, spy

def f(x):
    return (x[1:] - x[:-1]) ** 2

coloring = jacobian_coloring(f, input_shape=20)
spy(coloring, compressed=True)
```

Column compression (JVP/HVP) is shown side by side,
row compression (VJP) is shown stacked.

### Customizing Plots

`spy` accepts standard matplotlib arguments.
Pass an `ax` to plot on existing axes,
or use `markersize` and any `scatter` keyword argument:

```python
# mkdocs: render
import matplotlib.pyplot as plt
from asdex import jacobian_coloring, spy

def f(x):
    return (x[1:] - x[:-1]) ** 2

coloring = jacobian_coloring(f, input_shape=20)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
spy(coloring.sparsity, ax=ax1)
spy(coloring, ax=ax2)
ax1.set_title("Sparsity")
ax2.set_title("Coloring")
plt.tight_layout()
```

### Larger Patterns

For larger matrices, `spy` auto-scales the marker size:

```python
import jax.numpy as jnp
from asdex import hessian_coloring, spy

def rosenbrock(x):
    return jnp.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2)

coloring = hessian_coloring(rosenbrock, input_shape=100)
spy(coloring, compressed=True)
```

```python
# mkdocs: render
# mkdocs: hidecode
import jax.numpy as jnp
from asdex import hessian_coloring, spy

def rosenbrock(x):
    import jax.numpy as jnp
    return jnp.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2)

coloring = hessian_coloring(rosenbrock, input_shape=100)
spy(coloring, compressed=True)
```
