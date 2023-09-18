#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Utilities for configuration parsing
"""
import os
import shutil
import string
import random

import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler

import yaml
import json
from easydict import EasyDict
from pprint import pprint

from mmt import _repopath_ as mmt_repopath
from mmt.utils import dirs


def id_generator(size = 6, chars = string.ascii_lowercase + string.digits, forbidden = "_"):
    """Generate random strings of characters and digits than can be used as
    unique identifier
    
    
    Parameters
    ----------
    size: int
        Length of the returned string of characters
    
    chars: list
        Admissible characters. Default are lower-case alphanumeric ASCII characters
    
    
    Returns
    -------
    idstr: str
        String of `size` characters randomly taken among `chars`
    """
    
    idstr = "".join([random.choice(chars) for _ in range(size)])
    while forbidden in idstr:
        idstr = "".join([random.choice(chars) for _ in range(size)])
    
    return idstr

def setup_logging(log_dir):
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler(
        "{}exp_debug.log".format(log_dir), maxBytes=10**6, backupCount=5
    )
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler(
        "{}exp_error.log".format(log_dir), maxBytes=10**6, backupCount=5
    )
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, "r") as config_file:
        try:
            config_dict = json.load(config_file)
            # EasyDict allows to access dict values as attributes (works recursively).
            config = EasyDict(config_dict)
            if not config.data_folder.startswith("/"):
                config.data_folder = os.path.join(mmt_repopath, config.data_folder)
            return config, config_dict
        except ValueError:
            print("INVALID JSON file format.. Please provide a good json file")
            exit(-1)


def get_config_from_yaml(yaml_file):
    """
    Get the config from a YAML file
    :param yaml_file: the path of the config file
    :return: config(EasyDict)
    """
    with open(yaml_file, "r") as config_file:
        config = yaml.safe_load(config_file)
    
    if "[id]" in config["xp_name"]:
        config["xp_name"] = config["xp_name"].replace(
            "[id]", id_generator()
        )
    
    return EasyDict(config)

def get_config(config_file):
    """Parse the config from the given file"""
    assert os.path.isfile(config_file), f"No config file here: {config_file}"
    
    if config_file.endswith(".yaml"):
        config = get_config_from_yaml(config_file)
    elif config_file.endswith(".json"):
        config = get_config_from_json(config_file)
        config = convert_older_config(config)
    else:
        raise ValueError("Invalid config file. It must be a YAML or a JSON file")
    
    return config

def convert_older_config(old_config):
    """Ensure backward compatibility with JSON config files used in version < 0.2"""
    raise NotImplementedError

def process_config(config_file, quiet=False):
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
    try:
        config.xp_name
        if not quiet:
            print(" *************************************** ")
            print(f"The experiment name is {config.xp_name}")
            print(" *************************************** ")
    except AttributeError:
        print("ERROR!!..Please provide the xp_name in json file..")
        exit(-1)
    
    # for k, v in config.items():
        # if v == "None":
            # config[k] = None

    # Create directories to be used for that experiment
    config.paths.summary_dir = os.path.join(config.paths.experiments_dir, config.xp_name, "summaries")
    config.paths.checkpoint_dir = os.path.join(config.paths.experiments_dir, config.xp_name, "checkpoints")
    config.paths.out_dir = os.path.join(config.paths.experiments_dir, config.xp_name, "out")
    config.paths.log_dir = os.path.join(config.paths.experiments_dir, config.xp_name, "logs")
    dirs.create_dirs(
        [config.paths.summary_dir, config.paths.checkpoint_dir, config.paths.out_dir, config.paths.log_dir]
    )
    
    # Setup logging in the project
    setup_logging(config.paths.log_dir)
    
    shutil.copy(config_file, os.path.join(config.paths.log_dir, "config.yaml"))
    
    # if not quiet:
        # logging.getLogger().info("Hi, This is root.")
        # logging.getLogger().info(
            # "After the configurations are successfully processed and dirs are created."
        # )
        # logging.getLogger().info("The pipeline of the project will begin now.")
        # logging.getLogger().info(" THE Configuration of your experiment ..")
        # logging.getLogger().info(str(config))
    
    return config

