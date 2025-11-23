"""Logging configuration for StackAI Vector DB Server."""

import logging
import sys

from app.config import settings


def setup_logging() -> None:
    """Configure logging for the application.

    Uses DEBUG level if settings.debug is True, otherwise INFO.
    """
    level = logging.DEBUG if settings.debug else logging.INFO

    # Configure root logger for "app" namespace
    root_logger = logging.getLogger("app")
    root_logger.setLevel(level)

    # Avoid adding multiple handlers if called multiple times
    if root_logger.handlers:
        return

    # Console handler with formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Format: timestamp - level - message
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)
