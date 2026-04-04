"""
utils/logger.py
================
Centralized logging configuration using loguru.
"""

import sys
import os
from loguru import logger


def setup_logger(log_level: str = "INFO", log_file: str = None):
    """
    Configure loguru logger for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file:  Optional file path to write logs to
    """
    # Remove default handler
    logger.remove()

    # Console handler with colour
    logger.add(
        sys.stdout,
        level=log_level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
            "<level>{message}</level>"
        ),
    )

    # Optional file handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logger.add(
            log_file,
            level=log_level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} | {message}",
        )

    return logger
