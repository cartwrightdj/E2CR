import cv2
import matplotlib.pyplot as plt
import numpy as np

def getRowBoundries(line):
    """
    Compute the minimum and maximum y-coordinates for a given line.

    Args:
        line (list of tuples): A line represented by a list of (y, x) points.

    Returns:
        tuple: Minimum and maximum y-coordinates of the line.
    """
    if not line:
        raise ValueError("Line is empty")


    min_y = min(y for y, x in line)
    max_y = max(y for y, x in line)

    return min_y, max_y

def cropSegmentFromImage(image, lines, axis='y'):
    """
    Crop areas between each pair of generally horizontal or vertical lines,
    from the top to the first line, and from the last line to the bottom of the page.
    Return a list of cropped images with only the area between the lines included.

    Args:
        image (np.array): The input image.
        lines (list of list of tuples): List of lines, where each line is a list of (y, x) points.
        axis (str): Axis along which to perform the cropping ('x' or 'y').

    Returns:
        list of np.array: List of cropped images between the lines.
    """
    

    # Ensure the image is not empty
    if image is None or image.size == 0:
        
        raise ValueError("Input image is empty or invalid")

    cropped_images = []

    # Determine if the image is color or grayscale
    is_color = len(image.shape) == 3

    # Convert to grayscale if the input is a color image
    if is_color:
    
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    height, width = gray_image.shape
   

    for i in range(len(lines)):
        mask_pts = []
        if i == 0:
            if axis == 'y':
                #logger.debug(f"Processing from top of the image to line {i}")
                y_start = 0
                _, y_end = getRowBoundries(lines[i])
                mask_pts.extend([[0, 0]])
                mask_pts.extend([[x, y] for y, x in lines[i]])
                mask_pts.extend([[width, 0]])
            else:
                #logger.debug(f"Processing from left of the image to line {i}")
                x_start = 0
                x_end, _ = getRowBoundries(lines[i])
                mask_pts.extend([[0, 0]])
                mask_pts.extend([[x, y] for y, x in lines[i]])
                mask_pts.extend([[0, height]])
        elif i == len(lines) - 1:
            if axis == 'y':
                #logger.debug(f"Processing between line {i-1} and bottom of the image")
                _, y_start = getRowBoundries(lines[i-1])
                y_end = height
                mask_pts.extend([[x, y] for y, x in lines[i-1]])
                mask_pts.extend([[x, y] for y, x in lines[i]][::-1])
                mask_pts.extend([[width, height], [0, height]])
            else:
                #logger.debug(f"Processing between line {i-1} and right of the image")
                x_start, _ = getRowBoundries(lines[i-1])
                x_end = width
                mask_pts.extend([[x, y] for y, x in lines[i-1]])
                mask_pts.extend([[x, y] for y, x in lines[i]][::-1])
                mask_pts.extend([[width, 0], [width, height]])
        else:
            if axis == 'y':
               #logger.debug(f"Processing between line {i-1} and line {i}")
                y_start, _ = getRowBoundries(lines[i-1])
                _, y_end = getRowBoundries(lines[i])
                mask_pts.extend([[x, y] for y, x in lines[i-1]])
                mask_pts.extend([[x, y] for y, x in lines[i]][::-1])
            else:
                #logger.debug(f"Processing between line {i-1} and line {i}")
                x_start, _ = getRowBoundries(lines[i-1])
                x_end, _ = getRowBoundries(lines[i])
                mask_pts.extend([[x, y] for y, x in lines[i-1]])
                mask_pts.extend([[x, y] for y, x in lines[i]][::-1])

        # Create a mask for the area between the current line and the next line
        mask = np.zeros_like(gray_image, dtype=np.uint8)
        pts = np.array(mask_pts, dtype=np.int32)
        o_mask = cv2.fillPoly(mask, [pts], 255)
        
        # Apply the mask to the original image to extract the area between the lines
        masked_image = cv2.bitwise_and(o_mask, image, mask=mask)
        masked_image = cv2.bitwise_not(masked_image)
        masked_image = cv2.bitwise_and(masked_image, o_mask, mask=o_mask)
        masked_image = cv2.bitwise_not(masked_image)
        
        # Crop the bounding box of the masked region
        if axis == 'y':
            cropped_image = masked_image[y_start:y_end, 0:width]
           
        else:
            cropped_image = masked_image[0:height, x_start:x_end]
           

        # Ensure the cropped image is not empty
        if cropped_image.size > 0:
            cropped_images.append(cropped_image)
            
    return cropped_images
