
import cv2
import numpy as np
from scipy import stats
from scipy.stats import entropy
from common import logger
from imageman import is_grayscale

OKRED = '\033[31m'
OKCYAN = '\033[96m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'

def get_image_stats(image):

    """
    Calculate and print the mean, median, mode, standard deviation, variance, min, and max of the pixel values of an image.
    
    Parameters:
    image (np.ndarray): The input image (grayscale or color).
    
    Returns:
    stats_dict (dict): A dictionary containing the calculated statistics.
    """
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    height, width = image.shape[:2]
    # Flatten the image to 1D array of pixel values
    pixels = gray.flatten()

    # Calculate statistics
    mean_val = np.mean(pixels)
    median_val = np.median(pixels)
    mode_val = stats.mode(pixels, axis=None)
    std_val = np.std(pixels)
    var_val = np.var(pixels)
    min_val = np.min(pixels)
    max_val = np.max(pixels)

    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)

    # Calculate the entropy
    ientropy = entropy(hist, base=2)

    #ientropy = entropy(image, square(40))

    # Create a dictionary to hold the statistics
    stats_dict = {
        'dtype': image.dtype,
        'height': height,
        'width': width,
        'mean': mean_val,
        'median': median_val,
        'mode': mode_val,
        'std_dev': std_val,
        'variance': var_val,
        'min': min_val,
        'max': max_val,
        'entropy': ientropy
    }

    print("\n\033[31mImage Statistics")
    for stat_name, stat_value in stats_dict.items():
        print(f"{stat_name}: {stat_value}")

    print("\n\033[0m")

    return stats_dict

def shadeByDistanceBetweenInk(image: np.array, mode: str = 'both', text_color: str = 'black'): 
    logger.debug(f"Received image with shape: {image.shape}, mode:{mode}, text_color:{text_color}")

    if mode not in ['column', 'row', 'both']:
        logger.warning(f"Received incorrect mode: {mode}. Mode must be 'row', 'column', or 'both'")
        mode = 'both'

    if not is_grayscale(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.info("Converted to Grayscale")
        logger.debug(f"is_grayscale(ppImage={image.shape}) -> False")

    # Invert the image if the text color is black
    if text_color == 'black':
        image = np.bitwise_not(image)

    # Scale the image to 8-bit range
    height, width = image.shape[:2]
    value_inc = 255 / width
    cv2.imwrite('received.tiff', image)

    logger.debug(f"Calculating row shading")
    if mode == 'row' or mode == 'both':
        # Process rows
        for r in range(height):
            start_y = 0
            row_has_white_pixel = False
            for c in range(width):
                if image[r, c] == 255:  # White pixel
                    if start_y < c:
                        # Calculate the shading value proportional to the distance and image width
                        value = 255 - int((c - start_y) * value_inc)
                        image[r, start_y:c] = value
                    start_y = c  # Move to the pixel after the white pixel
                    row_has_white_pixel = True

            # Handle the case where the row ends without a trailing white pixel
            if row_has_white_pixel and start_y < width:
                value = 255 - int((width - start_y) * value_inc)
                image[r, start_y:width] = value

    if mode == 'column' or mode == 'both':
        value_inc = 255 / height
        # Process columns
        for c in range(width):
            start_x = 0
            col_has_white_pixel = False
            for r in range(height):
                if image[r, c] == 255:  # White pixel
                    if start_x < r:
                        # Calculate the shading value proportional to the distance and image height
                        value = 255 - int((r - start_x) * value_inc)
                        image[start_x:r, c] = value
                    start_x = r  # Move to the pixel after the white pixel
                    col_has_white_pixel = True

            # Handle the case where the column ends without a trailing white pixel
            if col_has_white_pixel and start_x < height:
                value = 255 - int((height - start_x) * value_inc)
                image[start_x:height, c] = value

    cv2.imwrite('shaded.tiff', image)

    # Invert the image back if the text color is black
    if text_color == 'black':
        image = np.bitwise_not(image)

    return image

def is_white_ink(image: np.ndarray) -> str:
    """
    Determine if the image has white text on a black background or black text on a white background.

    Parameters:
    - image: A numpy array representing the input image.

    Returns:
    - A string indicating the text and background color: 'white on black' or 'black on white'.
    """
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Calculate the number of white and black pixels
    num_white_pixels = np.sum(gray_image > 127)
    num_black_pixels = np.sum(gray_image <= 127)

    # Determine the predominant color scheme
    if num_white_pixels > num_black_pixels:
        return False
    else:
        return True