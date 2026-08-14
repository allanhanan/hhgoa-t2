"""Logging configuration for VoiceAgentRAG."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up and return the package logger."""
    logger = logging.getLogger("vor")
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the package namespace."""
    return logging.getLogger(f"vor.{name}")
