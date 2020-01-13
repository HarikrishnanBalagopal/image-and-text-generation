"""
Module for utility functions.
"""

import os
import errno
import datetime

def mkdir_p(path):
    """
    Make directory and all necessary parent directories given by the path.
    """

    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def get_timestamp():
    """
    Return the current date and time in year, month, day, hour, minute and second format
    """

    now = datetime.datetime.now()
    return now.strftime('%Y_%m_%d_%H_%M_%S')
