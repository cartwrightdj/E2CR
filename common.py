import os
import sys

import numpy as np
from loguru import logger
logger.remove()  # Remove the default logger
logger.add(sys.stderr, level="TRACE")

cwd = os.getcwd()

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKRED = '\033[31m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

CROP_VERTICLE = 1
CROP_HORIZONTAL = 0
COL_HIST = 0
ROW_HIST = 1

class E2CR:
    DENOIS =    0b0001
    THRESH =    0b0010
    ATHRESH =   0b0100
    BLUR =      0b1000

class DefaultSeg:
    EXPECTED_IMAGE_HW = 30

class PreProcessing:
    APPLY_ERODE = False
    erodeKernel = np.array([[0, 0, 0], [0, 3, 0], [0, 0, 0]],dtype=np.uint8)
    threshold = 180
    APPLY_ADAPTIVE_THRESHOLD = True
    APPLY_DILATION = False
    DILATE_KERNEL = (2,3)
    APPLY_DENOIS = True


def create_unique_filepath(path):
    """
    Ensure necessary folders are created for the given path. If it's a file path,
    handle the situation where the file already exists by appending a number at the end.

    Args:
        path (str): The desired file or folder path.

    Returns:
        str: The unique file path with necessary folders created.
    """
    # Check if the path is a file or a directory
    base, extension = os.path.splitext(path)
    if extension:  # It's a file path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # Check if file already exists and create a unique file path
        counter = 1
        new_filepath = path
        while os.path.exists(new_filepath):
            new_filepath = f"{base}_{counter}{extension}"
            counter += 1
        
        return new_filepath
    else:  # It's a directory path
        os.makedirs(path, exist_ok=True)
        return path

