"""
__author__ = "Hager Rady and Mo'men AbdelRazek"

Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""

import matplotlib
import argparse
from mmt.utils import config as utilconf
from mmt.agents import multiLULC
import os

def main():
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

    print(f"_____----- {os.environ.get('HSA_VISIBLE_DEVICES')} -----_____")
    print(f"_____----- {os.environ.get('HIP_VISIBLE_DEVICES')} -----_____")
    
    # Create the Agent and pass all the configuration to it then run it..
    agent_class = getattr(multiLULC, config.agent.type)
    agent = agent_class(config, **config.agent.params)
    agent.run()
    agent.finalize()


if __name__ == '__main__':
    main()
