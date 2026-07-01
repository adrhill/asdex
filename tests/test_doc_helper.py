"""Tests for the shared-docstring interpolation helper ``_fill_doc``.

These pin the robustness contract of the ``{placeholder}`` substitution:
registered tokens are filled, ordinary braces are left alone, and an
unregistered token fails loudly (naming the function) instead of silently
corrupting the rendered docstring.
"""

import pytest

import asdex
from asdex._doc_helper import _FRAGMENTS, _fill_doc


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
