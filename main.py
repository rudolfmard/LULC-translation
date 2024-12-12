"""
__author__ = "Hager Rady and Mo'men AbdelRazek"

Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""

import os
import torch
import matplotlib
import argparse
import datetime
from mmt.utils import config as utilconf
from mmt.agents import multiLULC
import torch.distributed as dist

def main():
    # Set a custom timeout (30 minutes) since process 0 currently does evaluation alone and takes more time than the default.
    # using dist.barrier() only holds for the duration set here.
    dist.init_process_group(
        backend='nccl',
        init_method='env://',  # Required for torch.distributed.run
        timeout=datetime.timedelta(seconds=1800)
    )

    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    # Choose the appropriate backend (see: https://matplotlib.org/stable/users/explain/figure/backends.html)
    matplotlib.use("AGG")
    
    # Parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'config',
        metavar='config_yaml_file',
        default='None',
        help='The configuration file in YAML format')
    args = arg_parser.parse_args()
    
    # Parse the config json file
    config = utilconf.process_config(args.config)
    
    # Create the Agent and pass all the configuration to it then run it..
    agent_class = getattr(multiLULC, config.agent.type)
    agent = agent_class(config, **config.agent.params)
    agent.run()
    agent.finalize()


if __name__ == '__main__':
    main()
