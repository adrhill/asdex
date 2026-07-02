"""Tests for the shared-docstring interpolation helper ``_fill_doc``.

These pin the robustness contract of the ``{placeholder}`` substitution:
registered tokens are filled, ordinary braces are left alone, and an
unregistered token fails loudly (naming the function) instead of silently
corrupting the rendered docstring.
"""

import importlib.util
from pathlib import Path

import pytest

import asdex
from asdex._docstrings import _FRAGMENTS, _fill_doc


def test_fill_doc_substitutes_registered_placeholder():
    """A registered ``{placeholder}`` is replaced by its fragment."""

    @_fill_doc
    def f():
        """Doc with {has_aux} inside."""

    assert f.__doc__ == f"Doc with {_FRAGMENTS['has_aux']} inside."


def test_fill_doc_leaves_ordinary_braces_untouched():
    """Dict literals and numeric format specs pass through unchanged.

    This is the property ``str.format`` lacked: it treated every brace as a
    field and raised on the dict literal below.
    """

    @_fill_doc
    def f():
        """Return {"a": 1} formatted as {0:.2f} with an empty {} brace."""

    assert f.__doc__ == 'Return {"a": 1} formatted as {0:.2f} with an empty {} brace.'


def test_fill_doc_unknown_placeholder_raises_naming_the_function():
    """An unregistered placeholder raises at decoration time, naming the function."""
    with pytest.raises(KeyError, match=r"typo_key.*_bad_doc_fn"):

        @_fill_doc
        def _bad_doc_fn():
            """Doc referencing {typo_key}."""


def test_fill_doc_noop_without_docstring():
    """A function without a docstring is returned unchanged."""

    def f():
        return None

    assert _fill_doc(f) is f
    assert f.__doc__ is None


def test_public_api_docstrings_have_no_unrendered_fragment_tokens():
    """No public symbol's rendered docstring contains a literal registered token.

    A leftover ``{jit}`` / ``{f_jac}`` / ... would mean substitution silently
    failed, so their absence confirms every public docstring rendered fully.
    """
    tokens = [f"{{{key}}}" for key in _FRAGMENTS]
    offenders = {}
    for name in asdex.__all__:
        doc = getattr(asdex, name).__doc__ or ""
        present = [t for t in tokens if t in doc]
        if present:
            offenders[name] = present
    assert not offenders, f"Unrendered fragment tokens: {offenders}"


def test_griffe_extension_renders_fragments_for_static_docs():
    """The docs griffe extension resolves every fragment in a static load.

    ``_fill_doc`` runs at import time, but mkdocstrings reads docstrings
    statically off the AST and never imports the code, so it needs
    ``docs/_griffe_extensions.py`` to re-run the same substitution at build time.
    This reproduces that static load with the real extension and asserts no
    public docstring still carries a registered ``{placeholder}`` token, which is
    the exact failure that rendered raw ``{f_jac}`` / ``{jit}`` on the docs site.
    The runtime test above cannot catch it: it only sees the interpolated
    ``__doc__``, not what griffe reads from source.
    """
    griffe = pytest.importorskip("griffe")

    repo_root = Path(__file__).resolve().parents[1]
    ext_path = repo_root / "docs" / "_griffe_extensions.py"
    spec = importlib.util.spec_from_file_location("_asdex_griffe_ext", ext_path)
    assert spec is not None
    assert spec.loader is not None
    ext_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ext_module)

    loader = griffe.GriffeLoader(
        search_paths=[str(repo_root / "src")],
        extensions=griffe.load_extensions(ext_module.FillDocstrings()),
    )
    mod = loader.load("asdex")

    tokens = [f"{{{key}}}" for key in _FRAGMENTS]
    offenders = {}
    for name in asdex.__all__:
        obj = mod[name]
        doc = (obj.docstring.value if obj.docstring is not None else "") or ""
        present = [t for t in tokens if t in doc]
        if present:
            offenders[name] = present
    assert not offenders, f"Unrendered fragment tokens in static docs: {offenders}"
