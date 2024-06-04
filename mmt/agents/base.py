#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
"""

import logging


class BaseAgent:
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    
    Following the formatting of [this template](https://github.com/moemen95/Pytorch-Project-Template),
    the agent performs traing, validation and testing of the models.
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")

    def load_checkpoint(self, file_name) -> None:
        """Latest checkpoint loader


        Parameters
        ----------
        file_name: str
            Name of the checkpoint file
        """
        raise NotImplementedError

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0) -> None:
        """Checkpoint saver


        Parameters
        ----------
        file_name: str
            Name of the checkpoint file
        is_best: bool
            Flag to indicate whether current checkpoint's metric is the best so far
        """
        raise NotImplementedError

    def run(self) -> None:
        """The main operator"""
        raise NotImplementedError

    def train(self) -> None:
        """Main training loop"""
        raise NotImplementedError

    def train_one_epoch(self) -> None:
        """One epoch of training"""
        raise NotImplementedError

    def validate(self) -> None:
        """One cycle of model validation"""
        raise NotImplementedError

    def finalize(self) -> None:
        """Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader"""
        raise NotImplementedError
