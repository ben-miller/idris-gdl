"""Configuration management for the project."""

from lib.config.logging import configure_logging
from lib.config.training import TrainingConfig

__all__ = ["TrainingConfig", "configure_logging"]
