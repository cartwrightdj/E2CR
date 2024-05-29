import os
import sys

import numpy as np
from loguru import logger

DEBUG = True

logger.remove()  # Remove the default logger
if DEBUG:
    logger.add(sys.stderr, level="TRACE")
else:
    logger.add(sys.stderr, level="INFO")
log_file = "E2CR.log"
logger.add(os.path.join(".", 'debug', log_file), format="{time: >14.4f} | {function} | {level} | {message}", level="TRACE", rotation="5 MB", compression="zip")

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


class SEG_DEFAULT:
    EXPECTED_IMAGE_HW = 45
    THRESH_RATE = .96
    IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    OUTPUT_PATH = os.path.join(os.getcwd(),'output')
    INPUT_PATH = os.path.join(os.getcwd(),'sample_images_for_ocr','onetime')


class DEFAULT:
    EPS = 5

    DENOISE_FIRST = True

    THRESH_RATE = 96
    PEAK_DIST = None
    PEAK_PROM = None


pp_config = {
    'applyDenoise': True,                       # used by apply_denoising
    'useAdaptiveThreshold': True,               # used by apply_adaptive_threshold
    'applyErode': False,                        # used by apply_erosion
    'erodeKernel': np.ones((5, 5), np.uint8),   # used by apply_erosion
    'threshold': 128,                           # used by apply_simple_threshold
    'maxValue': 255,                            # used by apply_simple_threshold
    'applyDilation': False,                     # used by apply_dilation
    'dilateKernalSize': (5, 5),                 # used by apply_dilation
    'applyMorphology': False,                   # used by apply_morphology
    'h': 9,                                     # used by apply_denoising
    'tWindowSize': 5,                           # used by apply_denoising
    'sWindowSize': 5,                           # used by apply_denoising
    'adaptiveBlockSize': 21,                    # used by apply_adaptive_threshold
    'adaptiveC': 4,                             # used by apply_adaptive_threshold
    'minBlackAreaSize': 90,                     # used by remove_small_black_areas
    'morphKernelSize': (3, 3),                  # used by apply_morphology
    'rsb_threshold': 128,                       # used by remove_small_black_areas
    'rsb_maxValue': 255                         # used by remove_small_black_areas
}

seg_config = {
    'threshRate': 85,                           # ! Whole number % if the sum of pixels across and axis exceed is greater than threshRate % than the max of any in the same exis
    'eps': 10,                                  # int, the maximum distance between two samples for one to be considered as in the neighborhood of the other (DBSCAN parameter)     
    'min_samples': 1,                           # int, the number of samples (or total weight) in a neighborhood for a point to be considered as a core point (DBSCAN parameter)
    'prominence': None,
    'max_ibp_loss': 95,                          # maximum amount of points initail break point can filter without reverting the points found with the threshold
    'distance': 20
}


class RuntimeParameters:
    applyDenoise = None
    useAdaptiveThreshold = None
    applyErode = None
    erodeKernel = None
    threshold = None
    maxValue = None
    applyDilation = None
    dilateKernalSize = None
    applyMorphology = None
    h = None
    tWindowSize = None
    sWindowSize = None
    adaptiveBlockSize = None
    adaptiveC = None
    minBlackAreaSize = None
    morphKernelSize = None
    rsb_threshold = None
    rsb_maxValue = None

    threshRate = None
    eps = None
    min_samples = None
    prominence = None
    distance = None
    max_ibp_loss = None

    @staticmethod
    def update_parameters(config: dict):
        for key, value in config.items():
            setattr(RuntimeParameters, key, value)

    @staticmethod
    def display():
        params = RuntimeParameters.__dict__
        function_params = {
            "apply_denoising": ["applyDenoise", "h", "tWindowSize", "sWindowSize"],
            "apply_erosion": ["applyErode", "erodeKernel"],
            "apply_adaptive_threshold": ["useAdaptiveThreshold", "adaptiveBlockSize", "adaptiveC"],
            "apply_simple_threshold": ["threshold", "maxValue"],
            "apply_dilation": ["applyDilation", "dilateKernalSize"],
            "apply_morphology": ["applyMorphology", "morphKernelSize"],
            "remove_small_black_areas": ["minBlackAreaSize", "rsb_threshold", "rsb_maxValue"],
            "group_by_proximity": ["eps", "min_samples"],
            "findInitialBreakPoints": ["prominence",'distance','max_ibp_loss']
          

        }

        print(f"\n{Colors.OKBLUE}Runtime Parameters:")
        for func, keys in function_params.items():
            param_str = ', '.join([f"{key}={params[key]}" for key in keys if params[key] is not None])
            print(f"{func} parameters: {param_str}")

    @staticmethod
    def set_parameter(name: str, value: float):
        if hasattr(RuntimeParameters, name):
            setattr(RuntimeParameters, name, value)
        else:
            raise RuntimeParameters(f"Statistic '{name}' does not exist.")

class Statistics:
    fhp_loss = 0
    chp_loss = 0
    paths_found = 0
    text_rows = 0

    y_init_peaks = 0                        # number of candidate break points from initial filtering  
    y_pf_loss = 0                           # percent of break points lost to prominence filter
    y_ibp_loss = 0                          # percent n axis elements lost by initial filtering  
    y_df_loss = 0                        

    @staticmethod
    def display():
        params = Statistics.__dict__
        function_params = {
            "apply_denoising": ["text_rows"],
            "y_init_peaks": ["y_init_peaks"],
            "y_ibp_loss": ["y_ibp_loss"],
            "y_pf_loss": ["y_pf_loss"]
         }

        print(f"\nStatistics:")     
        for func, keys in function_params.items():
            param_str = ', '.join([f"{key}={params[key]}" for key in keys if params[key] is not None])
            print(f"Stat: {param_str}")

    @staticmethod
    def set_statistic(name: str, value: float):
        setattr(Statistics, name, value)





