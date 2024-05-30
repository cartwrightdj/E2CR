import os
import sys

import numpy as np
from loguru import logger

DEBUG = True
DEBUG_FOLDER ="E:/E2CR/debug"

logger.remove()  # Remove the default logger
if DEBUG:
    logger.add(sys.stderr, level="TRACE")
else:
    logger.add(sys.stderr, level="INFO")
log_file = "E2CR.log"
logger.add(os.path.join(".", 'debug', log_file), format="{time: >14.4f} | {function} | {level} | {message}", level="TRACE", rotation="10 MB", compression="zip")

cwd = os.getcwd()

class DefaultParameters:
    axis: str ='y'                              #default axis to operate in, most operations use 'y'
    # Pre Processing
    removeBorder: bool = True                   # attmpt to remove border from source image
    # Denoise
    applyDenoise = True                         # attmpt to remove noise from source image 
    h = 9
    tWindowSize = 5
    sWindowSize = 5
                            
    useAdaptiveThreshold = True                 # used by apply_adaptive_threshold    
    applyErode = False                           # used by apply_erosion
    erodeKernel = np.ones((5, 5), np.uint8)     # used by apply_erosion
    threshold = None                            # used by apply_erosion
    maxValue = None                             # used by apply_erosion
    applyDilation = False                        #used by apply_erosion
    dilateKernalSize = (5, 5)
    applyMorphology = False
    
    # AdaptiveShreshold
    adaptiveBlockSize = 21
    adaptiveC = 4 
    # Normal Threshold
    simple_threshold = 120
    simple_max_value = 255
    # Morphology
    morphKernelSize = (3, 3)
    # Remove Spackle
    max_area_size = 90                          #Threshold area to determine which black areas to remove. Any connected component (black area) with an area smaller than this threshold will be removed.
    rs_threshold_value = 128                        #The threshold value used to binarize the image. Pixels with a value greater than or equal to this value are set to 0 (black) and the rest to max_value (white) when using cv2.THRESH_BINARY_INV.
    rs_max_value = 255                              #The maximum value to use with the THRESH_BINARY_INV thresholding.
    connectivity = 8                             #Connectivity to use when finding connected components. 4 for 4-way connectivity, 8 for 8-way connectivity.       


    useCumulativeSum: bool = True              # will create a cumulative sum for pixels across image axis, to find high points for thresholding
    subOnDecrease = True
    threshRate = 95                              # used to find indexes for row/columns with the highest values (least black), will only use rows/colums over this % of the maximum row/column sum(or cumulative sum)
    seek_ibp_loss = None
    eps = 10
    min_samples = 1
    findPeaks = False
    prominence = None
    distance = None
    max_threshRate_loss = 95
    f_proximity = 5

    usefilterValsByProximity = True
    min_white_space = 15
    

    method='y_max'

    # Path Finding
    log_cost_factor = 15
    bias_factor = 10
    gradient_actor = 5

    def __init__(self, **kwargs):
        # Set instance variables based on class-level defaults or provided values
        for attr in self.default_attributes():
            setattr(self, attr, kwargs.get(attr, getattr(self.__class__, attr)))
    
    @classmethod
    def default_attributes(cls):
        """
        Return a list of default attribute names.
        
        Returns:
            list: A list of default attribute names.
        """
        return [attr for attr in cls.__dict__ if not callable(getattr(cls, attr)) and not attr.startswith("__")]
    
    def print_instance_variables(self):
        """
        Print each instance variable and its value.
        """
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")

    @classmethod
    def print_defaults(cls):
        """
        Print each class-level default variable and its value.
        """
        for attr, value in cls.__dict__.items():
            if not callable(value) and not attr.startswith("__"):
                print(f"{attr}: {value}")



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

