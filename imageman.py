import cv2
import numpy as np
from common import *
from scipy.stats import kurtosis, skew, mode
from skimage.measure import shannon_entropy
from loguru import logger
from segmentation import getLineStats

def calculate_average_pixel_value(image):
    """
    Calculate the average pixel value of the entire image.

    Args:
        image (np.array): The input image.

    Returns:
        int: The average pixel value.
    """
    return int(np.mean(image))

def remove_border(image):
    """
    Replace irregular black border from an image with the average pixel value of the whole image using histograms of pixel sums.

    Args:
        image (np.array): The input image.

    Returns:
        np.array: The image with the black border replaced, retaining the same color channels.
    """
    logger.info("Starting border replacement using histogram method")

    # Ensure the image is not empty
    if image is None or image.size == 0:
        logger.critical("Input image is empty or invalid")
        raise ValueError("Input image is empty or invalid")

    # Check if the input is a color image and keep the original
    is_color = len(image.shape) == 3
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if is_color else image

    # Calculate the average pixel value of the entire image
    average_pixel_value = calculate_average_pixel_value(gray_image)
    logger.debug(f"Average pixel value: {average_pixel_value}")

    # Calculate the sum of pixels for each row and column
    row_sums = np.sum(gray_image, axis=1)
    col_sums = np.sum(gray_image, axis=0)
    logger.debug("Calculated row and column pixel sums")

    # Find the rows and columns where the sums exceed a threshold
    row_threshold = np.mean(row_sums) * 0.5
    col_threshold = np.mean(col_sums) * 0.5
    logger.debug(f"Row threshold: {row_threshold}, Column threshold: {col_threshold}")

    rows_with_content = np.where(row_sums > row_threshold)[0]
    cols_with_content = np.where(col_sums > col_threshold)[0]

    if len(rows_with_content) == 0 or len(cols_with_content) == 0:
        logger.warning("No significant content found, returning the original image")
        return image  # No significant content found, return the original image

    # Determine the bounding box of the content
    y_min, y_max = rows_with_content[0], rows_with_content[-1]
    x_min, x_max = cols_with_content[0], cols_with_content[-1]
    logger.info(f"Bounding box - y_min: {y_min}, y_max: {y_max}, x_min: {x_min}, x_max: {x_max}")

    # Create a mask for the border
    mask = np.ones_like(gray_image, dtype=np.uint8) * 255
    mask[y_min:y_max+1, x_min:x_max+1] = 0

    # Replace the border with the average pixel value
    border_indices = np.where(mask == 255)
    gray_image[border_indices] = average_pixel_value

    if is_color:
        # If the image was originally in color, apply the changes to each channel
        for i in range(3):
            channel = image[:, :, i]
            channel[border_indices] = average_pixel_value
            image[:, :, i] = channel
        return image
    else:
        return gray_image
    
def remove_small_black_areas(image, area_threshold=100):
    """
    Remove small black areas surrounded by white in an image.

    Args:
        image (np.array): The input image.
        area_threshold (int): Threshold area to determine which black areas to remove.

    Returns:
        np.array: The processed image with small black areas removed.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy.ndarray")

    # Convert to grayscale if the input is a color image
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to create a binary image
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Create an output image initialized to white
    output_image = np.ones_like(image) * 255
    
    # Loop through the components
    for i in range(1, num_labels):  # Skip the background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= area_threshold:
            # Keep the component if its area is larger than the threshold
            output_image[labels == i] = 0
    
    # Invert the image back to original format (black text on white background)
    #output_image = cv2.bitwise_not(output_image)
    
    return output_image

def draw_line(image, position, orientation='vertical', color=(0, 255, 0), thickness=2):
    """
    Draw a vertical or horizontal line on an image.

    Arguments:
    image -- np.ndarray, the input image
    position -- int, the position of the line (x-coordinate for vertical, y-coordinate for horizontal)
    orientation -- str, 'vertical' or 'horizontal' indicating the orientation of the line
    color -- tuple, the color of the line in BGR format (default is green)
    thickness -- int, the thickness of the line (default is 2)

    Returns:
    image_with_line -- np.ndarray, the image with the line drawn
    """
    # Copy the input image to avoid modifying the original
    image_with_line = image.copy()

    # Draw the line based on the specified orientation
    if orientation == 'vertical':
        cv2.line(image_with_line, (position, 0), (position, image.shape[0]), color, thickness)
    elif orientation == 'horizontal':
        cv2.line(image_with_line, (0, position), (image.shape[1], position), color, thickness)
    else:
        raise ValueError("Orientation must be 'vertical' or 'horizontal'")
    
    return image_with_line

def draw_path_on_image(image, path, axis='y'):
    """
    Draw the shortest path on the image.

    Arguments:
    image -- np.ndarray, the original image
    path -- list of tuples, the sequence of points in the shortest path
    axis -- str, the axis along which to draw the path ('y' for rows, 'x' for columns)

    Returns:
    output_image -- np.ndarray, the image with the path drawn
    """
    
    if len(image.shape) == 2:  # If the image is grayscale
        logger.debug("Converting grayscale image to BGR")
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        output_image = image
    
    logger.debug(f"Path length: {len(path)}")
    
    for (cy, cx) in path:
        output_image[cy, cx] = [255, 0, 0]

    # Calculate max and min coordinates from the path
    if axis == 'y':
        max_coord = max(path, key=lambda x: x[0])[0]
        min_coord = min(path, key=lambda x: x[0])[0]
        logger.debug(f"Max Y: {max_coord}, Min Y: {min_coord}")
    else:
        max_coord = max(path, key=lambda x: x[1])[1]
        min_coord = min(path, key=lambda x: x[1])[1]
        logger.debug(f"Max X: {max_coord}, Min X: {min_coord}")

    # Annotate max and min coordinates
    height, width = output_image.shape[:2]
    
    if axis == 'y':
        # Find the first coordinate with y value equal to max_coord and swap the values inline
        coord1 = next((coord for coord in path if coord[0] == max_coord), None)
        coord1 = (coord1[1] + 3, coord1[0]) if coord1 else None  # Swap and add 3 to the x value

        # Find the first coordinate with y value equal to min_coord and swap the values inline
        coord2 = next((coord for coord in path if coord[0] == min_coord), None)
        coord2 = (coord2[1] + 3, coord2[0]) if coord2 else None  # Swap and add 3 to the x value

        # Add text annotations to the output image if path is not empty
        if path:
            if coord1:
                cv2.putText(output_image, f"Max Y: {max_coord}", coord1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.BLUE, 1)
                
            if coord2:
                cv2.putText(output_image, f"Min Y: {min_coord}", coord2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.BLUE, 1)
                
    else:
        # Find the first coordinate with x value equal to max_coord and swap the values inline
        coord1 = next((coord for coord in path if coord[1] == max_coord), None)
        coord1 = (coord1[0], coord1[1] + 3) if coord1 else None  # Add 3 to the y value

        # Find the first coordinate with x value equal to min_coord and swap the values inline
        coord2 = next((coord for coord in path if coord[1] == min_coord), None)
        coord2 = (coord2[0], coord2[1] + 3) if coord2 else None  # Add 3 to the y value

        # Add text annotations to the output image if path is not empty
        if path:
            if coord1:
                cv2.putText(output_image, f"Max X: {max_coord}", coord1, cv2.FONT_HERSHEY_SIMPLEX, 0.4, Colors.BLUE, 1)
                logger.trace(f"Annotated Max X at: {coord1}")
            if coord2:
                cv2.putText(output_image, f"Min X: {min_coord}", coord2, cv2.FONT_HERSHEY_SIMPLEX, 0.4, Colors.BLUE, 1)
                logger.trace(f"Annotated Min X at: {coord2}")

    
    return output_image

def cropRowsFromImage(image, lines, axis='y'):
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
    logger.info(f"Starting crop between {len(lines)} lines")

    # Ensure the image is not empty
    if image is None or image.size == 0:
        logger.critical("Input image is empty or invalid")
        raise ValueError("Input image is empty or invalid")

    cropped_images = []

    # Determine if the image is color or grayscale
    is_color = len(image.shape) == 3

    # Convert to grayscale if the input is a color image
    if is_color:
        logger.debug("Converting color image to grayscale")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    height, width = gray_image.shape
    logger.debug(f"Image dimensions: height={height}, width={width}")

    for i in range(len(lines)):
        mask_pts = []
        if i == 0:
            if axis == 'y':
                #logger.debug(f"Processing from top of the image to line {i}")
                y_start = 0
                _, y_end = getLineStats(lines[i])
                mask_pts.extend([[0, 0]])
                mask_pts.extend([[x, y] for y, x in lines[i]])
                mask_pts.extend([[width, 0]])
            else:
                #logger.debug(f"Processing from left of the image to line {i}")
                x_start = 0
                x_end, _ = getLineStats(lines[i])
                mask_pts.extend([[0, 0]])
                mask_pts.extend([[x, y] for y, x in lines[i]])
                mask_pts.extend([[0, height]])
        elif i == len(lines) - 1:
            if axis == 'y':
                #logger.debug(f"Processing between line {i-1} and bottom of the image")
                _, y_start = getLineStats(lines[i-1])
                y_end = height
                mask_pts.extend([[x, y] for y, x in lines[i-1]])
                mask_pts.extend([[x, y] for y, x in lines[i]][::-1])
                mask_pts.extend([[width, height], [0, height]])
            else:
                #logger.debug(f"Processing between line {i-1} and right of the image")
                x_start, _ = getLineStats(lines[i-1])
                x_end = width
                mask_pts.extend([[x, y] for y, x in lines[i-1]])
                mask_pts.extend([[x, y] for y, x in lines[i]][::-1])
                mask_pts.extend([[width, 0], [width, height]])
        else:
            if axis == 'y':
               #logger.debug(f"Processing between line {i-1} and line {i}")
                y_start, _ = getLineStats(lines[i-1])
                _, y_end = getLineStats(lines[i])
                mask_pts.extend([[x, y] for y, x in lines[i-1]])
                mask_pts.extend([[x, y] for y, x in lines[i]][::-1])
            else:
                #logger.debug(f"Processing between line {i-1} and line {i}")
                x_start, _ = getLineStats(lines[i-1])
                x_end, _ = getLineStats(lines[i])
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
            logger.trace(f"Crop from y_start={y_start}, to y_end={y_end}")
        else:
            cropped_image = masked_image[0:height, x_start:x_end]
            logger.trace(f"Crop from x_start={x_start}, to x_end={x_end}")

        # Ensure the cropped image is not empty
        if cropped_image.size > 0:
            cropped_images.append(cropped_image)
        else:
            logger.warning(f"Empty crop region: {('y_start' if axis == 'y' else 'x_start')}={y_start if axis == 'y' else x_start}, {('y_end' if axis == 'y' else 'x_end')}={y_end if axis == 'y' else x_end}")

    logger.info("Finished cropping between lines")
    Statistics.text_rows = len(cropped_images)
    return cropped_images

def cropTextFromRow(image: np.ndarray, transitions: list[int]) -> list[np.ndarray]:
    """
    Crop the image at the given x transitions and return the cropped images as a list.

    Args:
        image (np.ndarray): The input binary image.
        transitions (List[int]): List of x-values where black pixels start or stop.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    height, width = image.shape
    cropped_images = []

    logger.info("Starting cropTextFromRow")
    logger.trace(f"cropTextFromRow(image.shape={image.shape}, transitions={transitions})")

    # Ensure transitions include the start and end of the image
    if 0 not in transitions:
        transitions.insert(0, 0)
    if width - 1 not in transitions:
        transitions.append(width - 1)

    logger.debug(f"Adjusted transitions: {transitions}")

    for i in range(len(transitions) - 1):
        start = transitions[i]
        end = transitions[i + 1]

        # Ensure valid cropping indices
        if start >= end or start < 0 or end > width:
            logger.warning(f"Invalid cropping indices: start={start}, end={end}")
            continue

        cropped_image = image[:, start:end]
        cropped_images.append(cropped_image)
        logger.debug(f"Cropped image from {start} to {end}, resulting shape: {cropped_image.shape}")

    logger.info(f"Finished cropping, produced {len(cropped_images)} cropped images")
    
    return cropped_images

def preProcessImage(ppImage, pp_config: dict, imageDesc: str = "") -> np.ndarray:
    """
    Preprocess an image with optional denoising, adaptive thresholding, erosion, dilation, and morphology.

    Arguments:
    ppImage -- np.ndarray, the input image
    pp_config -- dict, configuration parameters for preprocessing
    imageDesc -- str, additional description for debug images

    Returns:
    ppImage -- np.ndarray, the preprocessed image
    """
    try:
        logger.info("Starting image preprocessing")
        logger.debug(f"Configuration: {pp_config}")
        logger.trace(f"preProcessImage(ppImage={ppImage.shape}, pp_config={pp_config}, imageDesc={imageDesc})")

        step = 0

        # Update runtime parameters based on pp_config
        RuntimeParameters.update_parameters(pp_config)

        # Check and convert to grayscale if necessary
        if not is_grayscale(ppImage):
            ppImage = cv2.cvtColor(ppImage, cv2.COLOR_BGR2GRAY)
            logger.info("Converted to Grayscale")
            logger.trace(f"is_grayscale(ppImage={ppImage.shape}) -> False")

        # Ensure the image is of type uint8
        ppImage = ppImage.astype(np.uint8)

        # Detect and Remove Border
        step += 1
        ppImage = remove_border(ppImage)
        save_debug_image(ppImage, step, "Border_Removed", imageDesc)
        
        # Apply denoising
        if RuntimeParameters.applyDenoise:
            ppImage = apply_denoising(ppImage, step, imageDesc)

        # Apply erosion
        if RuntimeParameters.applyErode:
            ppImage = apply_erosion(ppImage, RuntimeParameters.erodeKernel, step, imageDesc)

        # Apply adaptive thresholding or simple thresholding
        if RuntimeParameters.useAdaptiveThreshold:
            ppImage = apply_adaptive_threshold(ppImage, step, imageDesc)
        else:
            ppImage = apply_simple_threshold(ppImage, RuntimeParameters.threshold, RuntimeParameters.maxValue, step, imageDesc)

        # Remove small black areas
        if RuntimeParameters.minBlackAreaSize:
            ppImage = remove_small_black_areas(ppImage, RuntimeParameters.minBlackAreaSize)
            step += 1
            save_debug_image(ppImage, step, "SpotRemoval", imageDesc)

        # Apply dilation
        if RuntimeParameters.applyDilation:
            ppImage = apply_dilation(ppImage, RuntimeParameters.dilateKernalSize, step, imageDesc)

        # Apply morphological operations
        if RuntimeParameters.applyMorphology:
            ppImage = apply_morphology(ppImage, step, imageDesc)
        
        logger.info("Finished image preprocessing")
        return ppImage
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

def is_grayscale(image):
    logger.trace(f"is_grayscale(image.shape={image.shape})")
    return len(image.shape) == 2

def apply_denoising(image, step, imageDesc):
    logger.info("Applying Denoising to Image")
    logger.trace(f"apply_denoising(image.shape={image.shape}, step={step}, imageDesc={imageDesc})")
    logger.debug(f"h={RuntimeParameters.h}, Template Window Size: {RuntimeParameters.tWindowSize}, Search Window Size: {RuntimeParameters.sWindowSize}")
    image = cv2.fastNlMeansDenoising(image, None, RuntimeParameters.h, RuntimeParameters.tWindowSize, RuntimeParameters.sWindowSize)
    save_debug_image(image, step, "Denoised_Image", imageDesc)
    return image

def apply_erosion(image, kernel, step, imageDesc):
    logger.info("Applying Erosion")
    logger.trace(f"apply_erosion(image.shape={image.shape}, kernel={kernel.shape}, step={step}, imageDesc={imageDesc})")
    image = cv2.erode(image, kernel)
    save_debug_image(image, step, "Eroded_Image", imageDesc)
    return image

def apply_adaptive_threshold(image, step, imageDesc):
    logger.info("Using Adaptive Thresholding")
    logger.trace(f"apply_adaptive_threshold(image.shape={image.shape}, step={step}, imageDesc={imageDesc})")
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, RuntimeParameters.adaptiveBlockSize, RuntimeParameters.adaptiveC)
    save_debug_image(image, step, "AdaptiveThreshold", imageDesc)
    return image

def apply_simple_threshold(image, threshold, max_value, step, imageDesc):
    logger.info("Using Simple Thresholding")
    logger.trace(f"apply_simple_threshold(image.shape={image.shape}, threshold={threshold}, max_value={max_value}, step={step}, imageDesc={imageDesc})")
    _, image = cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY)
    save_debug_image(image, step, "SimpleThreshold", imageDesc)
    return image

def apply_dilation(image, kernel_size, step, imageDesc):
    logger.info("Applying Dilation")
    logger.trace(f"apply_dilation(image.shape={image.shape}, kernel_size={kernel_size}, step={step}, imageDesc={imageDesc})")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    image = cv2.dilate(image, kernel, iterations=1)
    save_debug_image(image, step, "Dilated_Image", imageDesc)
    return image

def apply_morphology(image, step, imageDesc):
    logger.info("Applying Morphological Operations")
    logger.trace(f"apply_morphology(image.shape={image.shape}, step={step}, imageDesc={imageDesc})")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, RuntimeParameters.morphKernelSize)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    save_debug_image(image, step, "Morphology", imageDesc)
    return image

def remove_small_black_areas(image, area_threshold=100):
    """
    Remove small black areas surrounded by white in an image.

    Args:
        image (np.array): The input image.
        area_threshold (int): Threshold area to determine which black areas to remove.

    Returns:
        np.array: The processed image with small black areas removed.
    """
    logger.trace(f"remove_small_black_areas(image.shape={image.shape}, area_threshold={area_threshold})")
    
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy.ndarray")

    # Convert to grayscale if the input is a color image
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to create a binary image
    _, binary = cv2.threshold(image, RuntimeParameters.rsb_threshold, RuntimeParameters.rsb_maxValue, cv2.THRESH_BINARY_INV)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Create an output image initialized to white
    output_image = np.ones_like(image) * 255
    
    # Loop through the components
    for i in range(1, num_labels):  # Skip the background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= area_threshold:
            # Keep the component if its area is larger than the threshold
            output_image[labels == i] = 0
    
    return output_image

def save_debug_image(image, step, step_description, imageDesc):
    logger.trace(f"save_debug_image(image.shape={image.shape}, step={step}, step_description={step_description}, imageDesc={imageDesc})")
    debug_path = os.path.join(os.getcwd(), "debug", "PreProcess", f"Step_{step}_{step_description}_{imageDesc}.png")
    cv2.imwrite(debug_path, image)
    logger.debug(f"Saved {step_description} image to {debug_path}")

def percentWhite(image: np.ndarray) -> float:
    """
    Calculate the percentage of white pixels in the image.

    Args:
        image (np.ndarray): The input binary image.

    Returns:
        float: Percentage of white pixels in the image.
    """
    # Binarize the image to ensure it contains only 0 and 255 pixel values
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Count the number of white pixels (255)
    white_pixel_count = np.sum(binary_image == 255)
    total_pixel_count = binary_image.size

    # Calculate the percentage of white pixels
    white_pixel_percentage = (white_pixel_count / total_pixel_count) * 100

    return white_pixel_percentage

def crop_image_to_content(image, buffer=2):
    """
    Crop the image to its content (black pixels) and add a buffer around the edges.

    Args:
        image : the input image.
        buffer (int): Number of pixels to add as a buffer around the content.

    Returns:
        np.ndarray: The cropped image.
    """
    
    if image is None:
        raise ValueError(f"Error: Unable to load image at")

    # Find the bounding box of the black pixels
    coords = cv2.findNonZero(255 - image)  # Invert the image to find black pixels
    x, y, w, h = cv2.boundingRect(coords)

    # Add buffer to the bounding box
    x_start = max(x - buffer, 0)
    y_start = max(y - buffer, 0)
    x_end = min(x + w + buffer, image.shape[1])
    y_end = min(y + h + buffer, image.shape[0])

    # Crop the image
    cropped_image = image[y_start:y_end, x_start:x_end]

    return cropped_image
