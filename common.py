import os
import sys

import numpy as np
from loguru import logger
import cv2

logger.remove()  # Remove the default logger
logger.add(sys.stderr, level="TRACE")
log_file = "E2CR.log"
logger.add(os.path.join(".", log_file), format="{time: >14.4f} | {level} | {message}", level="TRACE", rotation="100 MB", compression="zip")

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

class SEG_DEFAULT:
    EXPECTED_IMAGE_HW = 45
    THRESH_RATE = .96
    IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    OUTPUT_PATH = os.path.join(os.getcwd(),'output')
    INPUT_PATH = os.path.join(os.getcwd(),'sample_images_for_ocr','onetime')




class PreProcessing:
    APPLY_ERODE = False
    erodeKernel = np.array([[0, 0, 0], [0, 3, 0], [0, 0, 0]],dtype=np.uint8)
    #erodeKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    threshold = 180
    APPLY_ADAPTIVE_THRESHOLD = True
    APPLY_DILATION = False
    DILATE_KERNEL = (2,3)
    APPLY_DENOIS = True
    APPLY_MORPHOLOGY = False






