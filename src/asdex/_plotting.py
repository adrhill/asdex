"""Matplotlib visualizations for SparsityPattern and ColoredPattern.

Requires the ``matplotlib`` optional dependency::

    pip install asdex[matplotlib]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from asdex.pattern import ColoredPattern, SparsityPattern


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
    ax: Axes | None = None,
    markersize: float | None = None,
    compressed: bool = False,
    **kwargs: Any,
) -> Axes | tuple[Axes, Axes]:
    """Plot a sparsity pattern or colored pattern using matplotlib.

    For a ``SparsityPattern``, plots nonzeros as black markers.
    For a ``ColoredPattern``, plots nonzeros colored by their color assignment.

    When ``compressed=True`` on a ``ColoredPattern``,
    shows uncompressed and compressed patterns side by side
    (column compression) or stacked (row compression).

    Args:
        pattern: The sparsity or colored pattern to plot.
        ax: Matplotlib axes to plot on.
            If ``None``, creates a new figure.
            Ignored when ``compressed=True`` (creates its own subplots).
        markersize: Marker size in points².
            If ``None``, auto-scaled based on matrix dimensions.
        compressed: If ``True`` and ``pattern`` is a ``ColoredPattern``,
            show original and compressed patterns side by side.
        **kwargs: Extra keyword arguments passed to ``ax.scatter``.

    Returns:
        A single ``Axes`` when plotting one pattern,
        or a ``(left, right)`` / ``(top, bottom)`` tuple
        when ``compressed=True``.
    """
    from asdex.pattern import ColoredPattern  # noqa: PLC0415

    if isinstance(pattern, ColoredPattern):
        if compressed:
            return _spy_compressed(pattern, markersize=markersize, **kwargs)
        return _spy_colored(pattern, ax=ax, markersize=markersize, **kwargs)
    return _spy_sparsity(pattern, ax=ax, markersize=markersize, **kwargs)


def _spy_sparsity(
    pattern: SparsityPattern,
    *,
    ax: Axes | None = None,
    markersize: float | None = None,
    **kwargs: Any,
) -> Axes:
    """Plot a ``SparsityPattern`` with uniform markers."""
    plt = _import_matplotlib()
    if ax is None:
        _, ax = plt.subplots()

    if markersize is None:
        markersize = _auto_markersize(pattern.m, pattern.n)

    kwargs.setdefault("color", "black")
    kwargs.setdefault("marker", "s")
    kwargs.setdefault("edgecolors", "none")

    ax.scatter(pattern.cols, pattern.rows, s=markersize, **kwargs)
    _format_axes(ax, pattern.m, pattern.n)
    return ax


def _spy_colored(
    colored: ColoredPattern,
    *,
    ax: Axes | None = None,
    markersize: float | None = None,
    **kwargs: Any,
) -> Axes:
    """Plot a ``ColoredPattern`` with markers colored by color assignment."""
    plt = _import_matplotlib()
    if ax is None:
        _, ax = plt.subplots()

    sp = colored.sparsity
    if markersize is None:
        markersize = _auto_markersize(sp.m, sp.n)

    # Map each nonzero to its color
    if colored._compresses_columns:
        c = colored.colors[sp.cols]
    else:
        c = colored.colors[sp.rows]

    kwargs.setdefault("marker", "s")
    kwargs.setdefault("edgecolors", "none")

    cmap = kwargs.pop("cmap", _discrete_cmap(colored.num_colors))
    ax.scatter(sp.cols, sp.rows, c=c, cmap=cmap, s=markersize, **kwargs)
    _format_axes(ax, sp.m, sp.n)
    return ax


def _spy_compressed(
    colored: ColoredPattern,
    *,
    markersize: float | None = None,
    **kwargs: Any,
) -> tuple[Axes, Axes]:
    """Plot uncompressed and compressed patterns side by side or stacked."""
    from asdex._display import _compressed_pattern  # noqa: PLC0415

    plt = _import_matplotlib()
    compressed = _compressed_pattern(colored)

    if colored._compresses_columns:
        _, (ax_left, ax_right) = plt.subplots(1, 2)
    else:
        _, (ax_left, ax_right) = plt.subplots(2, 1)

    _spy_colored(colored, ax=ax_left, markersize=markersize, **kwargs)
    _spy_colored_on(compressed, colored, ax=ax_right, markersize=markersize, **kwargs)

    ax_left.set_title(f"{colored.sparsity.m}×{colored.sparsity.n}")
    ax_right.set_title(f"{compressed.m}×{compressed.n}")
    plt.tight_layout()
    return ax_left, ax_right


def _spy_colored_on(
    pattern: SparsityPattern,
    colored: ColoredPattern,
    *,
    ax: Axes,
    markersize: float | None = None,
    **kwargs: Any,
) -> Axes:
    """Plot a compressed ``SparsityPattern`` using colors from a ``ColoredPattern``.

    The compressed pattern has different dimensions than the original,
    but its nonzeros inherit their color from the coloring.
    """
    if markersize is None:
        markersize = _auto_markersize(pattern.m, pattern.n)

    # Each entry (i, c) in the compressed pattern has color c
    c = pattern.cols if colored._compresses_columns else pattern.rows

    kwargs.setdefault("marker", "s")
    kwargs.setdefault("edgecolors", "none")

    cmap = kwargs.pop("cmap", _discrete_cmap(colored.num_colors))
    ax.scatter(pattern.cols, pattern.rows, c=c, cmap=cmap, s=markersize, **kwargs)
    _format_axes(ax, pattern.m, pattern.n)
    return ax


def _format_axes(ax: Axes, m: int, n: int) -> None:
    """Configure axes for sparsity plot (origin at top-left, tight limits)."""
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(m - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")


def _auto_markersize(m: int, n: int) -> float:
    """Compute a reasonable marker size based on matrix dimensions."""
    # Target: markers should fill ~70% of a cell in a 6-inch figure
    fig_size = 6.0
    dpi = 72.0
    cell_pts = (fig_size * dpi) / max(m, n)
    # scatter markersize is in points², so square it
    return max(0.5, (0.7 * cell_pts) ** 2)


def _discrete_cmap(n: int) -> Any:
    """Pick a colormap for ``n`` categorical colors.

    Uses ``tab10`` for up to 10 colors, ``tab20`` otherwise.
    Colors cycle when ``n`` exceeds 20.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if n <= 10:
        return plt.colormaps["tab10"]
    return plt.colormaps["tab20"]
