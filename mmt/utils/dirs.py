import os
import logging


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return:
    """
    #try:
    print("Create new direcories for results...")
    print("Currend working dir: ", os.getcwd())
    for dir_ in dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)
    print("Directories created!")
    #except Exception as err:
    #    logging.getLogger("Dirs Creator").info(
    #        "Creating directories error: {0}".format(err)
    #    )
    #    exit(-1)
