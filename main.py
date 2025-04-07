"""
__author__ = "Hager Rady and Mo'men AbdelRazek"

Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""

import os
import psutil
import torch
import matplotlib
import argparse
import datetime
from mmt.utils import config as utilconf
from mmt.agents import multiLULC
import torch.distributed as dist

# Copied from LUMI AI Guide:
# https://github.com/Lumi-supercomputer/LUMI-AI-Guide/blob/main/5-multi-gpu-and-node/ddp_visualtransformer.py
# The performance of the CPU mapping needs to be tested
def set_cpu_affinity(rank, local_rank):
    LUMI_GPU_CPU_map = {
        # A mapping from GCD to the closest CPU cores in a LUMI-G node
        # Note that CPU cores 0, 8, 16, 24, 32, 40, 48, 56 are reserved for the
        # system and not available for the user
        # See https://docs.lumi-supercomputer.eu/hardware/lumig/
        0: [49, 50, 51, 52, 53, 54, 55],
        1: [57, 58, 59, 60, 61, 62, 63],
        2: [17, 18, 19, 20, 21, 22, 23],
        3: [25, 26, 27, 28, 29, 30, 31],
        4: [1, 2, 3, 4, 5, 6, 7],
        5: [9, 10, 11, 12, 13, 14, 15],
        6: [33, 34, 35, 36, 37, 38, 39],
        7: [41, 42, 43, 44, 45, 46, 47],
    }
    cpu_list = LUMI_GPU_CPU_map[local_rank]
    print(f"Rank {rank} (local {local_rank}) binding to cpus: {cpu_list}")
    psutil.Process().cpu_affinity(cpu_list)

def main():
    # Set a custom timeout (30 minutes) since process 0 currently does evaluation alone and takes more time than the default.
    # using dist.barrier() only holds for the duration set here.
    dist.init_process_group(
        backend='nccl',
        init_method='env://',  # Required for torch.distributed.run
        timeout=datetime.timedelta(seconds=1800)
    )

    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    set_cpu_affinity(rank, local_rank)

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
