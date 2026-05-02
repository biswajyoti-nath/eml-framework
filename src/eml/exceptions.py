"""Custom exceptions for the EML framework.

Raises explicit, typed errors rather than generic Python exceptions so callers
can catch specific failure modes (domain violations, unsupported expressions)
without accidentally swallowing unrelated errors.
"""


class DomainError(Exception):
    """Raised when an expression violates the positive-domain assumptions of the
    EML exp-log fragment.

    Examples
    --------
    - log argument is not provably positive
    - non-integer power base is not provably positive
    - multiplication operand is not provably positive
    """


class UnsupportedExpressionError(Exception):
    """Raised when an expression contains a function that cannot be represented
    in the EML exp-log fragment.

    The supported fragment covers: exp, log, Abs, Add, Mul, Pow, and derivatives
    thereof. Any other SymPy function (e.g. sin, cos, gamma) triggers this error.
    """
