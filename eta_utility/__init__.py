import logging
import sys


def get_logger(name=None, level=None):
    prefix = "eta_utility"

    if name is not None and name != prefix:
        # Child loggers
        name = ".".join((prefix, name))
        log = logging.getLogger(name)
    else:
        # Main logger (only add handler if it does not have one already)
        log = logging.getLogger(prefix)
        log.propagate = False
        if not log.hasHandlers():
            handler = logging.StreamHandler(stream=sys.stdout)
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(logging.Formatter(fmt="[%(levelname)s] %(message)s"))
            log.addHandler(handler)

    if level is not None:
        log.setLevel(int(level * 10))
    return log
