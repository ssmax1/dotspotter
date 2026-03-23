import logging
import sys

def setup_logger():
    logger = logging.getLogger("dotspotter")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "\033[1;36m[dotspotter]\033[0m %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger