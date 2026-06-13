# Provides a consistent logger for every module — same format, same output stream.

import logging
import sys


class AppLogger:
    """Configures and returns a named logger with a consistent format across all modules."""

    FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATE_FORMAT = "%H:%M:%S"

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Creates a logger for the given module name — safe to call multiple times, won't duplicate handlers."""
        logger = logging.getLogger(name)

        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                AppLogger.FORMAT,
                datefmt=AppLogger.DATE_FORMAT,
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        return logger


def get_logger(name: str) -> logging.Logger:
    """Module-level shortcut so every file can just call get_logger(__name__)."""
    return AppLogger.get_logger(name)