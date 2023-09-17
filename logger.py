import logging


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    # Create handlers
    c_handler = logging.StreamHandler()

    # Create formatters and add them to handlers
    c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)

    return logger
