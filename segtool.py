import cv2
import numpy as np 
from common import logger
from imageman import is_grayscale

def sumColRows(image):
    """ 6/9/2024
    Sums the pixel values along the x and y axes of an image.

    Parameters:
    image (numpy array): The input image, which can be in grayscale or color.

    Returns:
    tuple: Two numpy arrays containing the sum of pixel values along the y-axis (columns) and x-axis (rows).
    """
    # Convert the image to grayscale if it is not already
    height, width = image.shape[:2]
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Sum the pixel values along the y-axis (sum along columns)
    col_sums = np.sum(gray, axis=0)

    # Sum the pixel values along the x-axis (sum along rows)
    row_sums = np.sum(gray, axis=1)

    return col_sums, row_sums

def firstContact(image: np.array):
    logger.debug(f"First Contact")
    if not is_grayscale(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.info("Converted to Grayscale")
        logger.debug(f"is_grayscale(ppImage={image.shape}) -> False")  
   
    height, width = image.shape[:2]
   
    first_contact=[] 

        # Process rows
    for r in range(height):
        start_y = 0
        row_has_white_pixel = False
        for c in range(width):
           
            if image[r, c] <= 5:  
                first_contact.append(c)
                break
            elif c == width-1:
                #first_contact.append(c)
                pass
    print(first_contact)
    return first_contact