import logging
import logging.handlers
import os
from typing import Optional

import config


def get_logger(name: Optional[str] = None) -> logging.Logger:
    os.makedirs(config.LOG_DIR, exist_ok=True)
    logger = logging.getLogger(name if name else __name__)

    if getattr(logger, "_initialized", False):
        return logger

    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=os.path.join(config.LOG_DIR, "trading_bot.log"), when="midnight", backupCount=7
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger._initialized = True  # type: ignore[attr-defined]

    return logger