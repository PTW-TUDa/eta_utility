from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator


@contextmanager
def suppress_logging(logger_name: str | None = None, level: int = logging.CRITICAL) -> Generator:
    logger = logging.getLogger(logger_name)
    current_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(current_level)
