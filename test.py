import cv2
from imageman import preProcessImage, cropRowsFromImage, draw_path_on_image, cropTextFromRow, percentWhite, crop_image_to_content
import numpy as np
import os

def cumulative_sum_image(image):
    """
    Create a color image where each pixel represents the cumulative sum of all previous pixels in the source image,
    distributed evenly across the three color channels. Also, create a matrix with the row number and the final sum for each row.
    
    Args:
        image (np.ndarray): The input image (grayscale or color).
        
    Returns:
        np.ndarray: The output color image with cumulative sums distributed across channels.
        np.ndarray: The matrix with row numbers and final cumulative sums.
    """
    # Convert to grayscale if the image is not already
    if len(image.shape) == 3:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        grayscale_image = image
    else:
        raise ValueError("Input image must be either a grayscale or a color image")
    
    # Invert the grayscale image
    #grayscale_image = cv2.bitwise_not(grayscale_image)
    
    # Compute the maximum sum a row could have
    height, width = grayscale_image.shape
    max_sum = width * 255
    
    # Thresholds for each channel
    threshold1 = max_sum / 3
    threshold2 = 2 * threshold1
    
    # Initialize the cumulative sum and color image
    color_image = np.zeros((*grayscale_image.shape, 3), dtype=np.uint8)
    cumulative_sums = []

    for row in range(height):
        cumulative_sum = 0
        for col in range(width):
            pixel_value = grayscale_image[row, col]
            cumulative_sum += pixel_value
            if cumulative_sum <= threshold1:
                color_image[row, col, 0] = int((cumulative_sum / threshold1) * 255)
            elif cumulative_sum <= threshold2:
                color_image[row, col, 0] = 255
                color_image[row, col, 1] = int(((cumulative_sum - threshold1) / threshold1) * 255)
            else:
                color_image[row, col, 0] = 255
                color_image[row, col, 1] = 255
                color_image[row, col, 2] = int(((cumulative_sum - threshold2) / threshold1) * 255)
        cumulative_sums.append([row, cumulative_sum])
    
    cumulative_sums_matrix = np.array(cumulative_sums, dtype=np.int64)
    
    return color_image, cumulative_sums_matrix



# Paths
image_path = 'E:/E2CR/sample_images_for_ocr/R. 317 (7).jpg'


def getTextFromImage(image_path: str):
    imageToSegment = cv2.imread(image_path,)
    ci, csm = cumulative_sum_image(imageToSegment )

    cv2.imwrite(os.path.join(os.getcwd(), 'debug','segmentation', 'cumSum.jpg'), ci)

    print(csm)
    print(max(csm[1]))
 



getTextFromImage(image_path)

