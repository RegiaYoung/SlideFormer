# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import functools
import logging
import sys
import os

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class LoggerFactory:

    @staticmethod
    def create_logger(name=None, level=logging.INFO):
        """create a logger

        Args:
            name (str): name of the logger
            level: level of logger

        Raises:
            ValueError is name is None
        """

        if name is None:
            raise ValueError("name for logger cannot be None")

        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] "
                                      "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s")

        logger_ = logging.getLogger(name)
        logger_.setLevel(level)
        logger_.propagate = False
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger_.addHandler(ch)
        return logger_


logger = LoggerFactory.create_logger(name="DeepSpeed", level=logging.INFO)


@functools.lru_cache(None)
def warning_once(*args, **kwargs):
    """
    This method is identical to `logger.warning()`, but will emit the warning with the same message only once

    Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the cache.
    The assumption here is that all warning messages are unique across the code. If they aren't then need to switch to
    another type of cache that includes the caller frame information in the hashing function.
    """
    logger.warning(*args, **kwargs)


logger.warning_once = warning_once


def print_configuration(args, name):
    logger.info("{}:".format(name))
    for arg in sorted(vars(args)):
        dots = "." * (29 - len(arg))
        logger.info("  {} {} {}".format(arg, dots, getattr(args, arg)))




def get_current_level():
    """
    Return logger's current log level
    """
    return logger.getEffectiveLevel()


def should_log_le(max_log_level_str):
    """
    Args:
        max_log_level_str: maximum log level as a string

    Returns ``True`` if the current log_level is less or equal to the specified log level. Otherwise ``False``.

    Example:

        ``should_log_le("info")`` will return ``True`` if the current log level is either ``logging.INFO`` or ``logging.DEBUG``
    """

    if not isinstance(max_log_level_str, str):
        raise ValueError(f"{max_log_level_str} is not a string")

    max_log_level_str = max_log_level_str.lower()
    if max_log_level_str not in log_levels:
        raise ValueError(f"{max_log_level_str} is not one of the `logging` levels")

    return get_current_level() <= log_levels[max_log_level_str]
