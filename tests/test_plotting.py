"""Smoke tests for matplotlib visualization."""

import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")

import matplotlib.pyplot as plt

from asdex import ColoredPattern, SparsityPattern, jacobian_coloring, spy

pytestmark = [pytest.mark.slow, pytest.mark.plot]


@pytest.fixture
def sparsity():
    """Tridiagonal 5x5 sparsity pattern."""
    return SparsityPattern.from_dense(
        np.array(
            [
                [1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1],
            ]
        )
    )


@pytest.fixture
def coloring():
    """Colored tridiagonal pattern."""

    def f(x):
        return (x[1:] - x[:-1]) ** 2

    return jacobian_coloring(f, 5)


class TestSpySparsityPattern:
    """Smoke tests for spy on SparsityPattern."""

    def test_returns_axes(self, sparsity):
        """Spy returns a matplotlib Axes."""
        ax = spy(sparsity)
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_custom_axes(self, sparsity):
        """Spy plots on a user-provided axes."""
        _fig, ax = plt.subplots()
        result = spy(sparsity, ax=ax)
        assert result is ax
        plt.close()

    def test_empty_pattern(self):
        """Spy handles empty patterns."""
        sp = SparsityPattern.from_coo([], [], (3, 4))
        ax = spy(sp)
        assert isinstance(ax, plt.Axes)
        plt.close()


class TestSpyColoredPattern:
    """Smoke tests for spy on ColoredPattern."""

    def test_returns_axes(self, coloring):
        """Spy returns a matplotlib Axes."""
        ax = spy(coloring)
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_compressed(self, coloring):
        """Spy with compressed=True returns an Axes."""
        ax = spy(coloring, compressed=True)
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_custom_axes(self, coloring):
        """Spy plots on a user-provided axes."""
        _fig, ax = plt.subplots()
        result = spy(coloring, ax=ax)
        assert result is ax
        plt.close()

    def test_custom_cmap_string(self, coloring):
        """Spy accepts a colormap name string."""
        ax = spy(coloring, cmap="viridis")
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_custom_cmap_object(self, coloring):
        """Spy accepts a colormap object."""
        ax = spy(coloring, cmap=plt.colormaps["viridis"])
        assert isinstance(ax, plt.Axes)
        plt.close()


class TestColormapCycling:
    """Test colormap cycling when num_colors exceeds cmap.N."""

    def test_discrete_no_cycling(self, coloring):
        """Discrete cmap with enough colors does not cycle."""
        ax = spy(coloring, cmap="tab10")
        data = np.asarray(ax.get_images()[0].get_array())
        valid = data[~np.isnan(data)]
        assert valid.max() < 10
        plt.close()

    def test_discrete_cycling(self):
        """Colors cycle when num_colors > cmap.N."""
        n = 12
        sp = SparsityPattern.from_coo(
            rows=np.arange(n, dtype=np.int32),
            cols=np.arange(n, dtype=np.int32),
            shape=(n, n),
        )
        colored = ColoredPattern(
            sparsity=sp,
            colors=np.arange(n, dtype=np.int32),
            num_colors=n,
            symmetric=False,
            mode="fwd",
        )

        ax = spy(colored, cmap="tab10")
        data = np.asarray(ax.get_images()[0].get_array())
        valid = data[~np.isnan(data)]
        # Colors 10 and 11 should cycle to 0 and 1
        assert valid.max() < 10
        plt.close()
