raise DeprecationWarning(f"{__name__}: This module is deprecated")
import logging
import os


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return:

    TODO: merge with mmt.utils.misc.create_directories and remove.
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as err:
        logging.getLogger("Dirs Creator").info(
            "Creating directories error: {0}".format(err)
        )
        exit(-1)
