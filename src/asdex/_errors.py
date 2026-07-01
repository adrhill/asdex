"""Errors and warnings raised across asdex."""


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
    """Raised when a user-supplied coloring violates a star-coloring constraint.

    See [`color_symmetric`][asdex.color_symmetric] with ``forced_colors``.
    """


class DenseColoringWarning(UserWarning):
    """Coloring uses as many colors as the dense baseline.

    Raised when sparse differentiation offers no speedup over dense differentiation.
    """
