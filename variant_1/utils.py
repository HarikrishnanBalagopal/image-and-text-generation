"""
Module for utility functions.
"""

import os
import errno

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
