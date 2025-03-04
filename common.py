import os
import sys

import numpy as np
from loguru import logger

DEBUG = True
DEBUG_FOLDER ="./debug"
VERBOSIY = 3

def logging(level: int,logfile="e2cr.log"):
    logger.remove()  # Remove the default logger
    VERBOSIY = level
    if DEBUG:
        match level:
            case 1:
                
                logger.add(sys.stderr, level="INFO")
                logger.info(f"{Colors.OKGREEN}Setting Verbose Level to 'INFO'{Colors.ENDC}")
            case 2:
                
                logger.add(sys.stderr, level="DEBUG")
                logger.info(f"{Colors.OKBLUE}Setting Verbose Level to 'DEBUG'{Colors.ENDC}")
            case 3:
                
                logger.add(sys.stderr, level="TRACE")
                logger.info(f"{Colors.OKRED}Setting Verbose Level to 'TRACE'{Colors.ENDC}")

    log_file = "E2CR.log"
    logger.add(os.path.join(".", 'debug', log_file), format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {function} | {level} | {message}", level="TRACE", rotation="10 MB", compression="zip")

cwd = os.getcwd()

FBCC_RECT =         0b00001
FBCC_AREA =         0b00010
CC_AREA_THRESH =    0b00100
CC_FRAME =          0b01000
FBCC_SIZE =           0b10000          


class Parameters:
    axis: str = 'y'  # Default axis to operate in, most operations use 'y'
    expectedTextRowHeight = 60
    
    class preProcessing:
        removeBorder: bool = True  # Attempt to remove border from source image
        applyDenoise = True  # Attempt to remove noise from source image 
        h = 9
        tWindowSize = 5
        sWindowSize = 5
        
        class thresHold:
            # Adaptive Threshold                        
            useAdaptiveThreshold = True  # Scanned images with variation in shading need to be adaptively thresholded 
            adaptiveBlockSize = None  # Left as None, settings will be made based on statistics of the image
            adaptiveC = None  

    applyErode = False  # Used by apply_erosion
    erodeKernel = np.ones((5, 5), np.uint8)  # Used by apply_erosion
    threshold = None  # Used by apply_erosion
    maxValue = None  # Used by apply_erosion
    applyDilation = False  # Used by apply_erosion
    dilateKernalSize = (5, 5)
    applyMorphology = False

    # Normal Threshold
    simple_threshold = 100
    simple_max_value = 255
    # Morphology
    morphKernelSize = (3, 3)
    # Remove Spackle
    max_area_size = 70  # Threshold area to determine which black areas to remove.
    rs_threshold_value = 128  # The threshold value used to binarize the image.
    rs_max_value = 255  # The maximum value to use with the THRESH_BINARY_INV thresholding.
    connectivity = 8  # Connectivity to use when finding connected components.

    useCumulativeSum: bool = False  # Will create a cumulative sum for pixels across image axis, to find high points for thresholding
    subOnDecrease = True
    
    class Segmentation:
        threshRate = 95  # Used to find indexes for row/columns with the highest values (least black)

        filterByDistance: bool = True
        filterByDistance_mode = 'both'   
    

    
    prominence = None
    distance = 30
    max_threshRate_loss = 95
    f_proximity = expectedTextRowHeight /3
    usefilterValsByProximity = True
    min_white_space = 15
    find_peaks_in_cluster_method = 'y_max'
    
    class pathFinder:
        log_cost_factor = 15
        bias_factor = 10
        gradient_factor = 5
        filterIndexByProximitySelector = 'v_max'
    
    class ccFilter:
        cc_threshold = 50
        cc_hw_ratio = 6
        cc_area_threshold = 3000 
        cc_area_threshold_ratio = .2
        cc_frame = 30

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
    def print_params(cls):
        """
        Print each class-level parameter and its value, including nested subclasses.
        """
        def print_class_params(c, prefix=''):
            # Print parameters for the given class
            for attr, value in c.__dict__.items():
                if not callable(value) and not str(attr).startswith("__"):
                    full_attr = f"{prefix}{attr}"
                    if value is None:
                        value_str = "None"
                    elif isinstance(value, (np.ndarray, tuple)):
                        value_str = ", ".join(map(str, value))
                    else:
                        value_str = str(value)
                    print(f"{full_attr}: {value_str}")
                
                # Recursively print nested classes
                if isinstance(value, type) and issubclass(value, object):
                    print_class_params(value, prefix=f"{attr}.")

        # Print parameters starting from the top-level class
        print_class_params(cls)
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

stats = []

class Statistics:
    fileName = None
    findPathStartPoints_loss =  None
    filterIndexByValThreshold_loss = None
    filterValsByProximity_loss = None
    filterByDistance_loss = None
    filterMinNConsecutive_loss = None
    CalculatedRowHight = None
                        

    @classmethod
    def display(cls):
        """
        Print each class-level parameter and its value.
        """
        # Find the maximum length of parameter names and values
        max_attr_length = max(len(attr) for attr in cls.__dict__.keys() if not callable(getattr(cls, attr)) and not str(attr).startswith("__"))
        values = [value for attr, value in cls.__dict__.items() if not callable(value) and not str(attr).startswith("__")]
        max_val_length = max(len(str(value)) if value is not None and not isinstance(value, (np.ndarray, tuple)) else len(", ".join(map(str, value))) if isinstance(value, (np.ndarray, tuple)) else 0 for value in values)

        # Print header
        print(f"{Colors.BOLD}{'Metric':<{max_attr_length}} {'Value':<{max_val_length}}{Colors.ENDC}")

        # Print each parameter and its value
        for attr, value in cls.__dict__.items():
            if not callable(value) and not str(attr).startswith("__"):
                attr_str = f"{attr:<{max_attr_length}}"
                value = str(value)
                if value is None:
                    value_str = "None"
                elif isinstance(value, (np.ndarray, tuple)):
                    value_str = ", ".join(map(str, value))
                else:
                    value_str = f"{value:<{max_val_length}}"
                print(f"{attr_str} {'.' * (max_val_length - len(value_str))} {value_str}")


    @staticmethod
    def set_statistic(name: str, value):
        attr = getattr(Statistics, name, None)
        if isinstance(attr, dict):
            # If the attribute is a dictionary, update it with the new entry
            attr.update(value)
        else:
            # Otherwise, set the attribute to the provided value
            setattr(Statistics, name, value)

    @classmethod
    def clear(cls):
        """
        Clear the data in the class by setting all class-level parameters to None.
        """
        for attr in cls.__dict__.keys():
            if not callable(getattr(cls, attr)) and not str(attr).startswith("__"):
                setattr(cls, attr, None)



