import logging


def get_logger(name, level="INFO"):
    logging.basicConfig(level=level)
    return logging.getLogger(name)
