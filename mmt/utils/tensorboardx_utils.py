raise DeprecationWarning(f"{__name__}: This module is deprecated")
import os
import subprocess
import time

from torch.utils.tensorboard import SummaryWriter


def tensorboard_summary_writer(config, comment):
    s = SummaryWriter(log_dir=config.paths.summary_dir, comment=comment)
    process = subprocess.Popen(
        ["tensorboard", "--logdir=" + config.paths.summary_dir],
        cwd=os.path.abspath(os.getcwd()),
    )
    time.sleep(10)
    return s, process
