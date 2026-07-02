"""Griffe extension bridging asdex's runtime docstring interpolation to mkdocstrings.

The runtime ``@_fill_doc`` decorator rewrites each public function's ``__doc__``
when ``asdex`` is imported (see ``src/asdex/_docstrings.py``).
mkdocstrings never imports the code:
its Python handler reads docstrings statically off the AST via griffe,
so it sees the raw ``{placeholder}`` tokens and renders them verbatim.

This extension re-runs the exact same substitution while griffe loads,
reusing ``asdex._docstrings`` as the single source of truth for the fragments
and the loud unknown-placeholder guard.
It is scoped to the functions the decorator targets (those carrying ``@_fill_doc``),
which griffe records statically,
so it touches exactly the runtime set and leaves every other docstring
(including ones that legitimately contain braces) untouched.
"""

from __future__ import annotations

from typing import Any

import griffe

from asdex._docstrings import _interpolate_fragments


def _has_fill_doc(func: griffe.Function) -> bool:
    """Whether ``func`` is decorated with ``@_fill_doc`` in the source."""
    return any(getattr(d.value, "name", None) == "_fill_doc" for d in func.decorators)


class FillDocstrings(griffe.Extension):
    def on_function_instance(self, *, func: griffe.Function, **kwargs: Any) -> None:
        if func.docstring is None or not _has_fill_doc(func):
            return
        func.docstring.value = _interpolate_fragments(
            func.docstring.value, where=func.path
        )
