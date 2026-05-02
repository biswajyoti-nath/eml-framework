"""Logging configuration for the EML framework.

Call ``configure_logging()`` once at your application entry point (e.g. in
``if __name__ == "__main__"`` blocks or experiment runners) to enable structured
log output. Individual modules obtain their logger via
``logging.getLogger(__name__)`` and never call this function themselves.
"""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a consistent format.

    Parameters
    ----------
    level:
        Logging level for the root logger (default: ``logging.INFO``).
        Use ``logging.DEBUG`` for verbose output during development.

    Notes
    -----
    This function is idempotent — calling it multiple times has no side effect
    beyond re-applying the same handler configuration.
    """
    logging.basicConfig(
        format="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
        force=True,
    )
