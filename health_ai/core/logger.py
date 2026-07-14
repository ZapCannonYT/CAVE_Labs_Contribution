"""
logger.py — Centralised logging configuration for Health AI v3.

Usage:
    from health_ai.core.logger import get_logger
    log = get_logger(__name__)
    log.info("Something happened")
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from health_ai.config.settings import LOG_DIR


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler — DEBUG and above (captures everything)
    # RotatingFileHandler prevents unbounded growth on long-running servers.
    log_file = LOG_DIR / "healthbot_v3.log"
    fh = RotatingFileHandler(
        log_file, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Error-only file handler
    err_file = LOG_DIR / "error.log"
    eh = RotatingFileHandler(
        err_file, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
    )
    eh.setLevel(logging.ERROR)
    eh.setFormatter(fmt)
    logger.addHandler(eh)

    logger.propagate = False
    return logger
