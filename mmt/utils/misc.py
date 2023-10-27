import time
import logging


def timeit(f):
    """Decorator to time Any Function"""

    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        seconds = end_time - start_time
        logging.getLogger("Timer").info(
            "   [-] %s : %2.5f sec, which is %2.5f min, which is %2.5f hour"
            % (f.__name__, seconds, seconds / 60, seconds / 3600)
        )
        return result

    return timed


def print_cuda_statistics():
    logger = logging.getLogger("Cuda Statistics")
    import sys
    from subprocess import call
    import torch

    logger.info("__Python VERSION:  {}".format(sys.version))
    logger.info("__pyTorch VERSION:  {}".format(torch.__version__))
    logger.info("__CUDA VERSION")
    try:
        call(["nvcc", "--version"])
    except:
        logger.info("nvcc not found")
    logger.info("__CUDNN VERSION:  {}".format(torch.backends.cudnn.version()))
    logger.info("__Number CUDA Devices:  {}".format(torch.cuda.device_count()))
    logger.info("__Devices")
    call(
        [
            "nvidia-smi",
            "--format=csv",
            "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free",
        ]
    )
    logger.info("Active CUDA Device: GPU {}".format(torch.cuda.current_device()))
    logger.info("Available devices  {}".format(torch.cuda.device_count()))
    logger.info("Current cuda device  {}".format(torch.cuda.current_device()))

def get_bbox_from_coord(x_coor, y_coor, patch_size, location="upper-left"):
    """Return a bounding box (xmin, ymin, xmax, ymax) from a single point coordinate.
    
    
    Parameters
    ----------
    x_coor: float
        Easting coordinate of the single point
    
    y_coor: float
        Northing coordinate of the single point
    
    patch_size: float
        Size of the bounding box, expressed as a difference of coordinate (xmax -xmin)
    
    location: {'lower-left', 'center', 'upper-right', 'upper-left'}
        Indicator to locate the single point coordinate w.r.t the bounding box.
        For example, if location='upper-left' (default), the single point is assumed
        to be located in the upper left corner of the bounding box that will be returned.
    
    
    Returns
    --------
    xmin, ymin, xmax, ymax: 4-tuple of float
        Bounding box (left, bottom, right, top)
    """
    if location == "lower-left":
        xmin, ymin, xmax, ymax = x_coor, y_coor, x_coor + patch_size, y_coor + patch_size
    elif location == "center":
        xmin, ymin, xmax, ymax = x_coor - patch_size/2, y_coor - patch_size/2, x_coor + patch_size/2, y_coor + patch_size/2
    elif location == "upper-right":
        xmin, ymin, xmax, ymax = x_coor - patch_size, y_coor - patch_size, x_coor, y_coor
    elif location == "upper-left":
        xmin, ymin, xmax, ymax = x_coor, y_coor - patch_size, x_coor + patch_size, y_coor
    else:
        raise ValueError(f"Unsupported location key: {location}. Supported keys are 'lower-left', 'center', 'upper-right', 'upper-left'.")
    
    return xmin, ymin, xmax, ymax

class InfiniIterTool:
    def __init__(self, start):
        self.i = 0
        self.size = len(start)
        self.start = iter(start)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        x = self.start[self.i % self.size]
        self.i += 1
        return x
