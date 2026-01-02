"""Logging configuration for the project."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure logging format for the entire project.

    Args:
        level: Logging level (default: logging.INFO)
    """
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(message)s",
        force=True,
    )
