import cv2
import numpy as np
from common import *
from scipy.stats import kurtosis, skew, mode
from skimage.measure import shannon_entropy
from loguru import logger
from segmentation import getLineStats
from PIL import Image
from PIL.ExifTags import TAGS

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
    
def removeSpackle(image, area_threshold=DefaultParameters.max_area_size):
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

def draw_path_on_image(image, path, axis='y',thickness=2):
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
    
    #for (cy, cx) in path:
    #    output_image[cy, cx] = [255, 0, 0]
    for (cy, cx) in path:
        cv2.circle(output_image, (cx, cy), thickness, (255, 0, 0), -1)

    # Calculate max and min coordinates from the path
    if axis == 'y':
        max_coord = max(path, key=lambda x: x[0])[0]
        min_coord = min(path, key=lambda x: x[0])[0]
        logger.trace(f"Max Y: {max_coord}, Min Y: {min_coord}")
    else:
        max_coord = max(path, key=lambda x: x[1])[1]
        min_coord = min(path, key=lambda x: x[1])[1]
        logger.trace(f"Max X: {max_coord}, Min X: {min_coord}")

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

 
def preProcessImage(ppImage, removeBorder: bool = DefaultParameters.removeBorder,
                    applyDenoise: bool = DefaultParameters.applyDenoise,
                    applyErode: bool = DefaultParameters.applyErode,
                    erodeKernel: bool = DefaultParameters.erodeKernel,
                    useAdaptiveThreshold: bool = DefaultParameters.useAdaptiveThreshold,
                    adaptiveBlockSize: int = DefaultParameters.adaptiveBlockSize,
                    adaptiveC: int = DefaultParameters.adaptiveC,
                    simple_threshold = DefaultParameters.simple_threshold,
                    simple_max_val = DefaultParameters.simple_max_value,
                    max_area_size = DefaultParameters.max_area_size,
                    rs_threshold_value = DefaultParameters.rs_threshold_value,
                    rs_max_value = DefaultParameters.rs_max_value,
                    applyDilation: bool = DefaultParameters.applyDilation,
                    dilateKernalSize = DefaultParameters.applyDilation,
                    applyMorphology: bool = DefaultParameters.applyMorphology) -> np.ndarray:
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
        logger.trace(f"preProcessImage Image)")

        step = 0

        # Check and convert to grayscale if necessary
        if not is_grayscale(ppImage):
            ppImage = cv2.cvtColor(ppImage, cv2.COLOR_BGR2GRAY)
            logger.info("Converted to Grayscale")
            logger.trace(f"is_grayscale(ppImage={ppImage.shape}) -> False")

        # Ensure the image is of type uint8
        ppImage = ppImage.astype(np.uint8)

        # Detect and Remove Border
        if removeBorder:
            step += 1
            ppImage = remove_border(ppImage)
            if DEBUG: save_debug_image(ppImage, "pp_Border_Removed")
        
        # Apply denoising
        if applyDenoise:
            ppImage = apply_denoising(ppImage)
            if DEBUG: save_debug_image(ppImage, "pp_Border_Removed")

        # Apply erosion
        if applyErode:
            step += 1
            ppImage = apply_erosion(ppImage, erodeKernel)

        # Apply adaptive thresholding or simple thresholding
        if useAdaptiveThreshold:
            step += 1
            ppImage = apply_adaptive_threshold(ppImage, adaptiveBlockSize,adaptiveC)
        else:
            step += 1
            ppImage = apply_simple_threshold(ppImage, simple_threshold, simple_max_val.maxValue)

        # Remove small black areas
        if max_area_size:
            ppImage = removeSpackle(ppImage, max_area_size, rs_threshold_value,rs_threshold_value )
            step += 1
            save_debug_image(ppImage, step, "SpotRemoval")

        # Apply dilation
        if applyDilation:
            ppImage = apply_dilation(ppImage, dilateKernalSize)

        # Apply morphological operations
        if applyMorphology:
            step += 1
            ppImage = apply_morphology(ppImage)
        
        logger.info("Finished image preprocessing")
        return ppImage
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

def is_grayscale(image):
    logger.trace(f"is_grayscale(image.shape={image.shape})")
    return len(image.shape) == 2

def apply_denoising(image,h = DefaultParameters.h,tWindowSize= DefaultParameters.tWindowSize,sWindowSize=DefaultParameters.sWindowSize ):
    logger.info("Applying Denoising to Image")
    logger.trace(f"PreProcessing - Apply Denoising(image.shape={image.shape}")
    logger.debug(f"h={h}, Template Window Size: {tWindowSize}, Search Window Size: {sWindowSize}")
    image = cv2.fastNlMeansDenoising(image, None, h, tWindowSize, sWindowSize)
    if DEBUG:save_debug_image(image,"pp_Denoised_Image")
    return image

def apply_erosion(image, kernel, step, imageDesc):
    logger.info("Applying Erosion")
    logger.trace(f"apply_erosion(image.shape={image.shape}, kernel={kernel.shape}, step={step}, imageDesc={imageDesc})")
    image = cv2.erode(image, kernel)
    return image

def apply_adaptive_threshold(image, adaptiveBlockSize = DefaultParameters.adaptiveBlockSize, adaptiveC = DefaultParameters.adaptiveC):
    logger.info("Using Adaptive Thresholding")
    logger.trace(f"apply_adaptive_threshold(image.shape={image.shape}, adaptiveBlockSize: {adaptiveBlockSize},  adaptiveC: {adaptiveC})")
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, adaptiveBlockSize, adaptiveC)
    return image

def apply_simple_threshold(image, simple_threshold = DefaultParameters.simple_threshold, simple_max_value = DefaultParameters.simple_max_value):
    logger.info("Using Simple Thresholding")
    logger.trace(f"apply_simple_threshold(image.shape={image.shape}, simple_threshold={simple_threshold}, max_value={simple_max_value})")
    _, image = cv2.threshold(image, simple_threshold, simple_max_value, cv2.THRESH_BINARY)
    return image

def apply_dilation(image, kernel_size):
    logger.info("Applying Dilation")
    logger.trace(f"apply_dilation(image.shape={image.shape}, kernel_size={kernel_size})")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    image = cv2.dilate(image, kernel, iterations=1)
    return image

def apply_morphology(image, morphKernelSize = DefaultParameters.morphKernelSize ):
    logger.info("Applying Morphological Operations")
    logger.trace(f"apply_morphology(image.shape={image.shape}, morphKernelSize: {morphKernelSize}")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morphKernelSize)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image

def removeSpackle(image, max_area_size=DefaultParameters.max_area_size, 
                  threshold_value=DefaultParameters.rs_threshold_value, 
                  max_value=DefaultParameters.rs_max_value, 
                  connectivity=DefaultParameters.connectivity):
    """
    Remove small black areas surrounded by white in an image.

    Args:
        image (np.array): The input image.
        area_threshold (int): Threshold area to determine which black areas to remove.
                              Any connected component (black area) with an area smaller
                              than this threshold will be removed.
        threshold_value (int): The threshold value used to binarize the image.
                              Pixels with a value greater than or equal to this value 
                              are set to 0 (black) and the rest to max_value (white) 
                              when using cv2.THRESH_BINARY_INV.
        max_value (int): The maximum value to use with the THRESH_BINARY_INV thresholding.
        connectivity (int): Connectivity to use when finding connected components. 
                            4 for 4-way connectivity, 8 for 8-way connectivity.

    Returns:
        np.array: The processed image with small black areas removed.
    """
    logger.trace(f"Starting removeSpackle(image.shape={image.shape}, area_threshold={max_area_size}, threshold_value={threshold_value}, max_value={max_value}, connectivity={connectivity})")

    # Ensure the input is a numpy array
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy.ndarray")

    # Convert to grayscale if the input is a color image
    if len(image.shape) == 3:
        logger.debug("Converting color image to grayscale.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to create a binary image (invert the binary image to get black areas as white)
    logger.debug("Applying binary thresholding.")
    _, binary = cv2.threshold(image, threshold_value, max_value, cv2.THRESH_BINARY_INV)

    # Find connected components
    logger.debug("Finding connected components.")
    # connectedComponentsWithStats finds all connected components in a binary image.
    # It returns the number of labels, the label matrix, the stats matrix, and the centroids matrix.
    # - num_labels: Number of labels (including background).
    # - labels: Label matrix where each connected component is assigned a unique label.
    # - stats: Statistics for each label, including bounding box and area (cv2.CC_STAT_*).
    # - centroids: Centroids of each connected component.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=connectivity)

    # Create an output image initialized to white
    output_image = np.ones_like(image) * 255

    # Loop through the components
    logger.debug(f"Processing {num_labels - 1} connected components.")
    rem_count = 0
    for i in range(1, num_labels):  # Skip the background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= max_area_size:
            # Keep the component if its area is larger than the threshold
            output_image[labels == i] = 0
        else:
            rem_count += 1

    logger.trace(f"Completed removeSpackle. removed {rem_count}")
    return output_image

def save_debug_image(image, step=0, Description='DEBUG'):
    debug_path = os.path.join(DEBUG_FOLDER, f"Step_{step}_{Description}.png")
    cv2.imwrite(debug_path, image)

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

def print_image_info(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Image not found or the path is incorrect: {image_path}")
        return

    # General image information
    print(f"\nImage Path: {image_path}")
    print(f"Image Shape: {image.shape}")
    print(f"Image Data Type: {image.dtype}")

    # Check the number of channels
    channels = image.shape[2] if len(image.shape) == 3 else 1
    print(f"Number of Channels: {channels}")

    # Convert the image to PIL format to extract metadata
    pil_image = Image.open(image_path)

    # Print image format
    print(f"Image Format: {pil_image.format}")

    # Extract and print EXIF metadata if available
    exif_data = pil_image._getexif()
    if exif_data is not None:
        print("EXIF Metadata:")
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            print(f"  {tag_name}: {value}")
    else:
        print("No EXIF metadata found.")

    # Print statistical information for each channel
    if channels > 1:
        channel_names = ['Blue', 'Green', 'Red']
        for i in range(channels):
            channel_data = image[:, :, i]
            print(f"\nChannel: {channel_names[i]}")
            print(f"  Min Value: {np.min(channel_data)}")
            print(f"  Max Value: {np.max(channel_data)}")
            print(f"  Mean Value: {np.mean(channel_data)}")
            print(f"  Standard Deviation: {np.std(channel_data)}")
    else:
        print("Image is grayscale.")
        print(f"  Min Value: {np.min(image)}")
        print(f"  Max Value: {np.max(image)}")
        print(f"  Mean Value: {np.mean(image)}")
        print(f"  Standard Deviation: {np.std(image)}")

def process_images_in_folder(folder_path):
    # Supported image extensions
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    # Iterate through all files in the folder
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                image_path = os.path.join(root, file)
                print_image_info(image_path)