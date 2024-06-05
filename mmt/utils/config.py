#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Utilities for configuration parsing
"""
import logging
import os
import shutil
from logging import Formatter
from logging.handlers import RotatingFileHandler
from pprint import pprint

import yaml
from easydict import EasyDict

from mmt import _repopath_ as mmt_repopath
from mmt.utils import misc


def setup_logging(log_dir) -> None:
    """Set up the logger (messaging format, file names and size...)

    Parameter
    --------
    log_dir: str
        Path to the directory where the log files will be stored
    """
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler(
        os.path.join(log_dir, "exp_debug.log"), maxBytes=10**6, backupCount=5
    )
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler(
        os.path.join(log_dir, "exp_error.log"), maxBytes=10**6, backupCount=5
    )
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)


def get_config(config_file) -> EasyDict:
    """Parse the config from the given file


    Parameter
    ---------
    config_file: str
        Path to the config file (YAML format)


    Returns
    -------
    config: EasyDict
        The configuration of the experiment
    """
    assert os.path.isfile(config_file), f"No config file here: {config_file}"
    assert config_file.endswith(
        ".yaml"
    ), "Config files other than YAML are no longer supported"

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if "[id]" in config["xp_name"]:
        config["xp_name"] = config["xp_name"].replace("[id]", misc.id_generator())

    return EasyDict(config)


def process_config(config_file, quiet=False) -> EasyDict:
    """Parse the configuration file, create experiment directories and set
    up logging for the whole program.


    Parameters
    ----------
    config_file: str
        Path of the config file

    quiet: bool, default=False
        Turn off all prints in this function


    Returns
    -------
    config: EasyDict
        The configuration of the experiment
    """
    config = get_config(config_file)
    if not quiet:
        print(" THE Configuration of your experiment ..")
        pprint(config)

    # Making sure that you have provided the xp_name.
    assert hasattr(config, "xp_name"), f"The config {config_file} has no xp_name"
    if not quiet:
        print(" *************************************** ")
        print(f"The experiment name is {config.xp_name}")
        print(" *************************************** ")

    # Create directories to be used for that experiment
    config.paths.summary_dir = os.path.join(
        config.paths.experiments_dir, config.xp_name, "summaries"
    )
    config.paths.checkpoint_dir = os.path.join(
        config.paths.experiments_dir, config.xp_name, "checkpoints"
    )
    config.paths.out_dir = os.path.join(
        config.paths.experiments_dir, config.xp_name, "out"
    )
    config.paths.log_dir = os.path.join(
        config.paths.experiments_dir, config.xp_name, "logs"
    )
    misc.create_directories(
        config.paths.summary_dir,
        config.paths.checkpoint_dir,
        config.paths.out_dir,
        config.paths.log_dir,
    )

    # Setup logging in the project
    setup_logging(config.paths.log_dir)

    shutil.copy(config_file, os.path.join(config.paths.log_dir, "config.yaml"))
    shutil.copy(
        config_file, os.path.join(config.paths.checkpoint_dir, "model_best.config.yaml")
    )

    return config
