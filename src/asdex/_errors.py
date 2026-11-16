"""Errors and warnings raised across asdex."""

_ISSUES_URL = "https://github.com/adrhill/asdex/issues"
"""Where users report missing handlers and bugs."""


class VerificationError(AssertionError):
    """Raised when asdex's sparse result does not match JAX's dense reference.

    This indicates that the detected sparsity pattern is missing nonzeros,
    which is a bug — asdex's patterns should always be conservative
    (i.e., contain at least all true nonzeros).
    If you encounter this error,
    please help out asdex's development by reporting this at
    https://github.com/adrhill/asdex/issues.
    """


class InvalidColoringError(ValueError):
    """Raised when a coloring violates the constraints it is checked against.

    See [`check_coloring_rows`][asdex.check_coloring_rows],
    [`check_coloring_cols`][asdex.check_coloring_cols],
    and [`check_coloring_symmetric`][asdex.check_coloring_symmetric].
    """


class DenseColoringWarning(UserWarning):
    """Coloring uses as many colors as the dense baseline.

    Raised when sparse differentiation offers no speedup over dense differentiation.
    """


class DensePatternWarning(UserWarning):
    """Detected sparsity pattern is fully dense.

    Emitted when sparsity detection finds no zeros,
    so sparse differentiation offers no speedup over dense differentiation.
    The pattern is still correct.

    The message tells two causes apart.
    If detection met primitives it has no handler for,
    it replaced them by conservative patterns, which may be what cost the sparsity.
    The message then lists the ones whose inputs depended on the function's inputs.
    Every unknown primitive is stored in
    [`SparsityPattern.fallbacks`][asdex.SparsityPattern].
    Please help out asdex's development by reporting them at
    <https://github.com/adrhill/asdex/issues>.
    Otherwise, the function may be dense by nature,
    or asdex's handlers may have over-approximated its sparsity.
    If you expected a sparser pattern, please report this too.
    """
