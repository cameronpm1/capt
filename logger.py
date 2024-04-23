import logging

LEVEL = None
FILTER = ""


FORMATS = {
    "DEBUG": logging.Formatter("[%(levelname)s]%(name)s.%(funcName)s(): %(message)s"),
    "INFO": logging.Formatter("[%(levelname)s]%(message)s"),
}


def getlogger(name):

    logger = logging.getLogger(name)

    # Set level
    if LEVEL is not None:
        level = getattr(logging, LEVEL)
        logger.setLevel(level)

        # Create handler
        # handler = logging.FileHandler(file, mode="w")
        handler = logging.StreamHandler()
        handler.setLevel(level)

        # Create formatter
        if LEVEL in FORMATS.keys():
            formatter = FORMATS[LEVEL]
            handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.addFilter(logging.Filter(FILTER))
    return logger