"""Namespace shim — import from ``eml`` directly, not from ``src``.

This file exists only to prevent accidental imports through the ``src``
namespace.  The installable package is ``eml`` (located at ``src/eml/``).

Usage::

    from eml import transform, eml, evaluate_eml  # correct
    from src.eml import ...                        # works but discouraged
"""

# Re-export nothing — users must import from `eml` directly.
