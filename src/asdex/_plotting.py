"""Matplotlib visualizations for SparsityPattern and ColoredPattern.

Requires the ``matplotlib`` optional dependency::

    pip install asdex[matplotlib]
"""

from __future__ import annotations

from typing import Any

import numpy as np

from asdex._pattern import ColoredPattern, SparsityPattern


def _import_matplotlib() -> Any:
    """Import matplotlib, raising a helpful error if not installed."""
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        msg = (
            "matplotlib is required for plotting. "
            "Install it with: pip install asdex[matplotlib]"
        )
        raise ImportError(msg) from None
    return plt


def spy(
    pattern: SparsityPattern | ColoredPattern,
    *,
    ax: Any = None,
    compressed: bool = False,
    cmap: Any = None,
    **kwargs: Any,
) -> Any:
    """Plot a sparsity pattern or colored pattern using matplotlib.

    For a ``SparsityPattern``, plots nonzeros as filled cells on a grid.
    For a ``ColoredPattern``, fills cells with their assigned color.

    When ``compressed=True`` on a ``ColoredPattern``,
    plots the compressed pattern after coloring instead of the original.

    Args:
        pattern: The sparsity or colored pattern to plot.
        ax: Matplotlib axes to plot on.
            If ``None``, creates a new figure.
        compressed: If ``True`` and ``pattern`` is a ``ColoredPattern``,
            plot the compressed pattern instead of the original.
        cmap: Matplotlib colormap for colored patterns.
            If ``None``, uses ``tab10``.
        **kwargs: Extra keyword arguments passed to ``ax.imshow``.

    Returns:
        The matplotlib axes with the plot.
    """
    if isinstance(pattern, ColoredPattern):
        return _spy_colored(pattern, ax=ax, compressed=compressed, cmap=cmap, **kwargs)
    return _spy_sparsity(pattern, ax=ax, **kwargs)


def _spy_sparsity(
    pattern: SparsityPattern,
    *,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Plot a ``SparsityPattern`` as a black-and-white grid."""
    from matplotlib.colors import ListedColormap  # noqa: PLC0415

    plt = _import_matplotlib()
    if ax is None:
        _, ax = plt.subplots()

    dense = pattern.todense().astype(float)
    bw_cmap = ListedColormap(["white", "black"])

    kwargs.setdefault("interpolation", "none")
    ax.imshow(dense, cmap=bw_cmap, vmin=0, vmax=1, **kwargs)
    _format_axes(ax, pattern.m, pattern.n)
    return ax


def _spy_colored(
    colored: ColoredPattern,
    *,
    ax: Any = None,
    compressed: bool = False,
    cmap: Any = None,
    **kwargs: Any,
) -> Any:
    """Plot a ``ColoredPattern`` with cells colored by color assignment."""
    plt = _import_matplotlib()
    if ax is None:
        _, ax = plt.subplots()

    if compressed:
        from asdex._display import _compressed_pattern  # noqa: PLC0415

        pattern = _compressed_pattern(colored)
    else:
        pattern = colored.sparsity

    # Build grid with NaN for zeros, color index for nonzeros
    grid = np.full((pattern.m, pattern.n), np.nan)
    if colored._compresses_columns:
        colors = pattern.cols if compressed else colored.colors[pattern.cols]
    else:
        colors = pattern.rows if compressed else colored.colors[pattern.rows]
    if cmap is None:
        cmap = "tab10"

    resolved_cmap = plt.colormaps[cmap] if isinstance(cmap, str) else cmap

    n_cmap = resolved_cmap.N
    is_discrete = n_cmap <= 20

    if colored.num_colors > n_cmap:
        # More colors than colormap entries: cycle
        grid[pattern.rows, pattern.cols] = colors % n_cmap
        vmax = n_cmap - 1
    elif is_discrete:
        # Discrete colormap: map index i to entry i
        grid[pattern.rows, pattern.cols] = colors
        vmax = n_cmap - 1
    else:
        # Continuous colormap: spread evenly for maximum contrast
        grid[pattern.rows, pattern.cols] = colors
        vmax = colored.num_colors - 1

    kwargs.setdefault("interpolation", "none")
    ax.imshow(grid, cmap=resolved_cmap, vmin=0, vmax=vmax, **kwargs)
    _format_axes(ax, pattern.m, pattern.n)
    return ax


def _format_axes(ax: Any, m: int, n: int) -> None:
    """Configure axes for sparsity plot (origin at top-left, tight limits)."""
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(m - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
