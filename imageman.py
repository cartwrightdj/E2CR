import cv2
import numpy as np
from common import *
from scipy.stats import kurtosis, skew, mode
#from skimage.measure import shannon_entropy
from loguru import logger
from PIL import Image
from PIL.ExifTags import TAGS
from utils import *
from skimage.util import view_as_blocks
from scipy.stats import entropy
import math

 
def pre_process_image(ppImage, 
                    fileHandle,
                    removeBorder: bool = Parameters.preProcessing.removeBorder,
                    applyDenoise: bool = Parameters.preProcessing.applyDenoise,
                    applyErode: bool = Parameters.applyErode,
                    erodeKernel: bool = Parameters.erodeKernel,
                    useAdaptiveThreshold: bool = Parameters.preProcessing.thresHold.useAdaptiveThreshold,
                    adaptiveBlockSize: int = Parameters.preProcessing.thresHold.adaptiveBlockSize,
                    adaptiveC: int = Parameters.preProcessing.thresHold.adaptiveC,
                    simple_threshold = Parameters.simple_threshold,
                    simple_max_val = Parameters.simple_max_value,
                    max_area_size = Parameters.max_area_size,
                    rs_threshold_value = Parameters.rs_threshold_value,
                    rs_max_value = Parameters.rs_max_value,
                    applyDilation: bool = Parameters.applyDilation,
                    dilateKernalSize = Parameters.applyDilation,
                    applyMorphology: bool = Parameters.applyMorphology) -> np.ndarray:
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
        logger.trace(f"preProcessImage: Handle:{fileHandle})")

        # Check and convert to grayscale if necessary
        if not is_grayscale(ppImage):
            ppImage = cv2.cvtColor(ppImage, cv2.COLOR_BGR2GRAY)
            logger.info("Converted to Grayscale")

        # Ensure the image is of type uint8
        ppImage = ppImage.astype(np.uint8)

        # Detect and Remove Border
        #if removeBorder:
        #    step += 1
        #    ppImage = remove_border(ppImage)
        #    if DEBUG: save_debug_image(ppImage, "pp_Border_Removed")

        if removeBorder:
            ppImage = crop_to_border(ppImage)
            if DEBUG: visual_debug(ppImage, action='save', operation_name=f"{fileHandle}_pp_crop_to_border")

         # Apply denoising
        if applyDenoise:
            ppImage_prior = ppImage
            ppImage = apply_denoising(ppImage)
            if DEBUG: 
                visual_debug(ppImage, action='save', operation_name=f"{fileHandle}_pp_denoised")
                visual_debug(util_img_diff(ppImage_prior, ppImage), operation_name=f"{fileHandle}_pp_denoised_diff")
            
           
        if useAdaptiveThreshold:              
            ppImage = apply_adaptive_threshold(ppImage, adaptiveBlockSize,adaptiveC)
            visual_debug(ppImage, action='save', operation_name=f"{fileHandle}_AdaptiveThreshold")
        else:
            ppImage = apply_simple_threshold(ppImage, simple_threshold, simple_max_val)
        #visual_debug(ppImage, action='save', operation_name=f"Image After Threshold (Adaptive={useAdaptiveThreshold}")
        
        '''
        ppImage, loss = filterByConectedComponents(ppImage,method=FBCC_RECT,hw_ratio=7)
        visual_debug(ppImage,action='save', operation_name=f"{fileHandle}_After Filter By CC")

        ppImage, loss2 = filterByConectedComponents(ppImage,method=FBCC_AREA, min_area=30)
        visual_debug(ppImage,action='save', operation_name=f"{fileHandle}_After Filter By CC_min")

        hm = util_img_cc_lbl_hm(ppImage)
        visual_debug(hm,action='save', operation_name="Contected Comp HeatMap 2,")

        ppImage , loss3 =filterByConectedComponents(ppImage,method=FBCC_SIZE, max_size=100000)
        visual_debug(ppImage,action='save', operation_name=f"{fileHandle}_After Filter By CC_max_size")

        Statistics.filterByConectedComponents_loss = [loss,loss2,loss3]
        #ppImage = filterByConectedComponents(ppImage,method=FBCC_AREA,max_area=3000)    
        '''   

        visual_debug(ppImage, operation_name=f"{fileHandle}_Before_LineRemoval")
        if False: # lines removal
            print("trying to remove")
            ppImage = remove_lines(ppImage)
        
        logger.info("Finished image preprocessing")
        if DEBUG: visual_debug(ppImage,action='save',operation_name=f"{fileHandle}_Image Pre-Processing Final Result")
        return ppImage
    except Exception as e:
        logger.error(f"Error during preprocessing: {e.with_traceback}")
        raise

def crop_to_border(image):
    """
    Crop the image to the border where significant content exists.

    Args:
        image (np.array): The input image.

    Returns:
        np.array: The cropped image containing only the significant content.
    """
    # Ensure the image is not empty
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or invalid")

    # Check if the input is a color image and keep the original
    is_color = len(image.shape) == 3
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if is_color else image

    # Calculate the sum of pixels for each row and column
    row_sums = np.sum(gray_image, axis=1)
    col_sums = np.sum(gray_image, axis=0)

    # Find the rows and columns where the sums exceed a threshold
    row_threshold = np.mean(row_sums) * 0.5
    col_threshold = np.mean(col_sums) * 0.5

    rows_with_content = np.where(row_sums > row_threshold)[0]
    cols_with_content = np.where(col_sums > col_threshold)[0]

    if len(rows_with_content) == 0 or len(cols_with_content) == 0:
        return image  # No significant content found, return the original image

    # Determine the bounding box of the content
    y_min, y_max = rows_with_content[0], rows_with_content[-1]
    x_min, x_max = cols_with_content[0], cols_with_content[-1]

    # Crop the image to the bounding box
    cropped_image = image[y_min:y_max, x_min:x_max]

    return cropped_image

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

def avgPixelValue(image):
    """
    Calculate the average pixel value of the entire image.

    Args:
        image (np.array): The input image.

    Returns:
        int: The average pixel value.
    """
    return int(np.mean(image))

def calculate_average_of_adjacent_pixels(image, y, x, direction='row'):
    """
    Calculate the average value of the 5 adjacent pixels in the specified direction.
    
    Args:
        image (np.array): The input image.
        y (int): The y-coordinate of the pixel.
        x (int): The x-coordinate of the pixel.
        direction (str): The direction to calculate the average ('row' or 'column').

    Returns:
        float: The average value of the 5 adjacent pixels.
    """
    if direction == 'row':
        adjacent_pixels = image[y, max(0, x-5):x] + image[y, x+1:min(image.shape[1], x+6)]
    elif direction == 'column':
        adjacent_pixels = image[max(0, y-5):y, x] + image[y+1:min(image.shape[0], y+6), x]

    return np.mean(adjacent_pixels)

def remove_border(image, side_threshold=200):
    """
    Replace irregular black border from an image with the average pixel value of the nearest 10 pixels within the specified threshold.

    Args:
        image (np.array): The input image.
        side_threshold (int): The number of pixels to consider from the sides for border detection.

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

    # Create a mask for the area outside the side_threshold
    mask = np.zeros_like(gray_image, dtype=np.uint8)
    mask[y_min:y_max+1, x_min:x_max+1] = 255

    # Replace the border with the average of the nearest 5 adjacent pixels in the row or column
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] == 0:
                if y < y_min or y > y_max or x < x_min or x > x_max:
                    if y < y_min:
                        avg_pixel_value = calculate_average_of_adjacent_pixels(gray_image, y_min, x, 'column')
                    elif y > y_max:
                        avg_pixel_value = calculate_average_of_adjacent_pixels(gray_image, y_max, x, 'column')
                    elif x < x_min:
                        avg_pixel_value = calculate_average_of_adjacent_pixels(gray_image, y, x_min, 'row')
                    elif x > x_max:
                        avg_pixel_value = calculate_average_of_adjacent_pixels(gray_image, y, x_max, 'row')
                    gray_image[y, x] = avg_pixel_value

    if is_color:
        # If the image was originally in color, apply the changes to each channel
        for i in range(3):
            channel = image[:, :, i]
            for y in range(mask.shape[0]):
                for x in range(mask.shape[1]):
                    if mask[y, x] == 0:
                        if y < y_min or y > y_max or x < x_min or x > x_max:
                            if y < y_min:
                                avg_pixel_value = calculate_average_of_adjacent_pixels(channel, y_min, x, 'column')
                            elif y > y_max:
                                avg_pixel_value = calculate_average_of_adjacent_pixels(channel, y_max, x, 'column')
                            elif x < x_min:
                                avg_pixel_value = calculate_average_of_adjacent_pixels(channel, y, x_min, 'row')
                            elif x > x_max:
                                avg_pixel_value = calculate_average_of_adjacent_pixels(channel, y, x_max, 'row')
                            channel[y, x] = avg_pixel_value
            image[:, :, i] = channel
        return image
    else:
        return gray_image

def draw_line(image, position, orientation='vertical', color=(0, 255, 0), thickness=1):
    """
    Draw a vertical or horizontal line on an image.

    Arguments:
    image -- np.ndarray, the input image
    position -- int, the position of the line (x-coordinate for vertical, y-coordinate for horizontal)
    orientation -- str, 'vertical' or 'horizontal' indicating the orientation of the line
    color -- tuple, the color of the line in BGR format (default is green)
    thickness -- int, the thickness of the line (default is 2)
from heapq import heappop, heappush
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
            logger.trace(f"Crop from y_start={y_start}, to y_end={y_end}")
        else:
            cropped_image = masked_image[0:height, x_start:x_end]
            logger.trace(f"Crop from x_start={x_start}, to x_end={x_end}")

        # Ensure the cropped image is not empty
        if cropped_image.size > 0:
            cropped_images.append(cropped_image)
        else:
            logger.warning(f"Empty crop region: {('y_start' if axis == 'y' else 'x_start')}={y_start if axis == 'y' else x_start}, {('y_end' if axis == 'y' else 'x_end')}={y_end if axis == 'y' else x_end}")

    logger.debug("Finished cropping between lines")
    logger.trace(f"Cropped {len(cropped_images)} images")
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

    logger.trace(f"cropTextFromRow(image.shape={image.shape}, transitions={transitions})")

    # Ensure transitions include the start and end of the image
    if 0 not in transitions:
        transitions.insert(0, 0)
    if width - 1 not in transitions:
        transitions.append(width - 1)

    for i in range(len(transitions) - 1):
        start = transitions[i]
        end = transitions[i + 1]

        # Ensure valid cropping indices
        if start >= end or start < 0 or end > width:
            logger.warning(f"Invalid cropping indices: start={start}, end={end}")
            continue

        cropped_image = image[:, start:end]
        cropped_images.append(cropped_image)
        logger.trace(f"Cropped image from {start} to {end}, resulting shape: {cropped_image.shape}")

    logger.trace(f"Finished cropping, produced {len(cropped_images)} cropped images")
    
    return cropped_images

def is_grayscale(image):
    logger.trace(f"is_grayscale(image.shape={image.shape})")
    return len(image.shape) == 2

def apply_denoising(image,h = Parameters.preProcessing.h,tWindowSize= Parameters.preProcessing.tWindowSize,sWindowSize=Parameters.preProcessing.sWindowSize ):
    logger.info("Applying Denoising to Image")
    logger.trace(f"PreProcessing - Apply Denoising(image.shape={image.shape}")
    logger.debug(f"h={h}, Template Window Size: {tWindowSize}, Search Window Size: {sWindowSize}")
    image = cv2.fastNlMeansDenoising(image, None, h, tWindowSize, sWindowSize)
    return image

def apply_erosion(image, kernel, step, imageDesc):
    logger.info("Applying Erosion")
    logger.trace(f"apply_erosion(image.shape={image.shape}, kernel={kernel.shape}, step={step}, imageDesc={imageDesc})")
    image = cv2.erode(image, kernel)
    return image

def calcBlockC(image):
    mean_intensity = np.mean(image)
    std_deviation = np.std(image)
    median_intensity = np.median(image)
    std_val = np.std(image)
    block_size = int(2 * (std_deviation // 2) + 1)  # Must be an odd number
    c_value = abs(mean_intensity - median_intensity)
    if c_value < 15: c_value = 15
    return block_size,  c_value

def apply_adaptive_threshold(image, adaptiveBlockSize = Parameters.preProcessing.thresHold.adaptiveBlockSize, adaptiveC = Parameters.preProcessing.thresHold.adaptiveC):
    logger.info("Using Adaptive Thresholding")
    if adaptiveBlockSize is None: 
        adaptiveBlockSize, _ = calcBlockC(image)
        logger.trace(f"Block Size Calculated as {adaptiveBlockSize}")

    if adaptiveC is None: 
        _, adaptiveC = calcBlockC(image)
        logger.trace(f"adaptiveC Calculated as {adaptiveC}")

    #adaptiveBlockSize, adaptiveC = calcBlockC(image)
    logger.trace(f"apply_adaptive_threshold(image.shape={image.shape}, adaptiveBlockSize: {adaptiveBlockSize},  adaptiveC: {adaptiveC})")
    
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, adaptiveBlockSize, adaptiveC)
    return image

def apply_simple_threshold(image, simple_threshold = Parameters.simple_threshold, simple_max_value = Parameters.simple_max_value):
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

def apply_morphology(image, morphKernelSize = Parameters.morphKernelSize ):
    logger.info("Applying Morphological Operations")
    logger.trace(f"apply_morphology(image.shape={image.shape}, morphKernelSize: {morphKernelSize}")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morphKernelSize)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image

def percentWhite(image: np.ndarray) -> float:
    """
    Calculate the percentage of white pixels in the image.

    Args:
        image (np.ndarray): The input binary image.

    Returns:
        float: Percentage of white pixels in the image.
    """
    # Binarize the image to ensure it contains only 0 and 255 pixel values
    #_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Count the number of white pixels (255)
    white_pixel_count = np.sum(image == 255)
    black_pixel_count = np.sum(image == 0)
    height, width = image.shape
    total_pixel_count = height * width
    #print(f"{height = }, {width = }, {white_pixel_count = }, {black_pixel_count =}, {total_pixel_count = }")

    # Calculate the percentage of white pixels
    white_pixel_percentage = (white_pixel_count / total_pixel_count) * 100

    return white_pixel_percentage

def crop_image_to_content(image, frame=2):
    """
    Crop the image to its content (black pixels) and add a frame around the edges.

    Args:
        image : the input image.
        frame (int): Number of pixels to add as a frame around the content.

    Returns:
        np.ndarray: The cropped image.
    """
    
    if image is None:
        raise ValueError(f"Error: Unable to load image at")

    # Find the bounding box of the black pixels
    coords = cv2.findNonZero(255 - image)  # Invert the image to find black pixels
    x, y, w, h = cv2.boundingRect(coords)

    # Add frame to the bounding box
    x_start = max(x - frame, 0)
    y_start = max(y - frame, 0)
    x_end = min(x + w + frame, image.shape[1])
    y_end = min(y + h + frame, image.shape[0])

    # Crop the image
    cropped_image = image[y_start:y_end, x_start:x_end]

    return cropped_image

def print_image_info(image_path):
    #Load the image using OpenCV
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

def visual_debug(dbg_image: np.array,axis='y',values=[], action='None', operation_name='DEBUG',alternate=False):
    logger.trace(f"Visual Debug: action:{action}")
    dbg_image = dbg_image.copy()
    
    if action == 'draw_lines':
        if len(dbg_image.shape) == 2:
            dbg_image = cv2.cvtColor(dbg_image.copy(), cv2.COLOR_GRAY2BGR)
        elif dbg_image.shape[2] == 3:
            dbg_image = dbg_image.copy()
        else:
            logger.warning(f"Could not saved debug image for {operation_name} to: {os.path.join(DEBUG_FOLDER, f'{operation_name}.tiff')}")

        c = Colors.RED
        for v in values:
            if axis == 'y':
                dbg_image = cv2.line(dbg_image, (0, int(v)), (dbg_image.shape[1], int(v)), color=c, thickness=2)
                cv2.putText(dbg_image, f"y:{v}", (25, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, Colors.BLUE, 2)
            else:
                dbg_image = cv2.line(dbg_image, (int(v), 0), (int(v), dbg_image.shape[0]), color=c, thickness=2)
                cv2.putText(dbg_image, f"x:{v}", (v + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, Colors.BLUE, 2)
            c = Colors.BLUE if c == Colors.RED else Colors.RED
            if alternate: c = Colors.BLUE if c == Colors.RED else Colors.RED
        cv2.imwrite(os.path.join(DEBUG_FOLDER, f'dbg_{operation_name}.tiff'), dbg_image)
        logger.trace(f"Saved debug image for {operation_name} to: {os.path.join(DEBUG_FOLDER, f'{operation_name}.tiff')}")

    if action == 'draw_paths':
        if len(dbg_image.shape) == 2:
            dbg_image = cv2.cvtColor(dbg_image.copy(), cv2.COLOR_GRAY2BGR)
        elif dbg_image.shape[2] == 3:
            dbg_image = dbg_image.copy()
        else:
            logger.warning(f"Could not saved debug image for {operation_name} to: {os.path.join(DEBUG_FOLDER, f'{operation_name}.tiff')}")

        for v in values:
            dbg_image = draw_path_on_image(dbg_image,v)
        save_results = cv2.imwrite(os.path.join(DEBUG_FOLDER, f'{operation_name}.tiff'), dbg_image) 
        if save_results:   
            logger.trace(f"Saved debug image for {operation_name} to: {os.path.join(DEBUG_FOLDER, f'{operation_name}.tiff')}")
        else:
            logger.warning(f"Failed to save debug image for {operation_name} to: {os.path.join(DEBUG_FOLDER, f'{operation_name}.tiff')}")

        dbg_v += 1
    
    if action == 'save':
        if not cv2.imwrite(os.path.join(DEBUG_FOLDER, f'{operation_name}.tiff'), dbg_image):
            logger.critical(f"Could not save: {os.path.join(DEBUG_FOLDER, f'{operation_name}.tiff'),}")

def filterByConectedComponents(image, method,
                               threshold=128,  # Default threshold value
                               hw_ratio=2.0,   # Default height-to-width ratio
                               min_area = None,
                               max_area = None,
                               max_size = None,
                               frame=10,       # Default frame size
                               area_threshold=0,
                               area_threshold_ratio=0.01):
    
    logger.debug(f"Filtering (Removing) Connected Components: method:{method}, threshold:{threshold}, hw_ratio:{hw_ratio}, frame:{frame}, area_threshold:{area_threshold}, area_threshold_ratio:{area_threshold_ratio}")

    # Convert to grayscale if the image is not already
    if len(image.shape) == 3 and image.shape[2] == 3:
        logger.warning("This function received a non-grayscale image, results will not be accurate")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = image.shape[:2]

    # Initialize the mask to keep valid connected components
    fcc_mask = np.zeros_like(image)
    fcc_mask[:] = 255

    # Thresholding to create a binary image
    _, binary_thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)

    # Connected component analysis
    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(binary_thresh)
    logger.trace(f"Found {num_labels} Connected Components")

    # Extract stats for connected components, excluding the background (label 0)
    x = stats[1:, cv2.CC_STAT_LEFT]
    y = stats[1:, cv2.CC_STAT_TOP]
    w = stats[1:, cv2.CC_STAT_WIDTH]
    h = stats[1:, cv2.CC_STAT_HEIGHT]
    area = stats[1:, cv2.CC_STAT_AREA]
    size = np.multiply(h, w)  # Correctly compute the size as an array

    # Calculate height-to-width ratio and determine valid labels
    hwratio = np.maximum(w / h, h / w)
    valid_labels = np.ones(num_labels - 1, dtype=bool)

    if method & CC_FRAME == CC_FRAME:
        valid_labels &= (x >= frame) & (x + w <= width - frame) & (y >= frame) & (y + h <= height - frame)

    if method & FBCC_RECT == FBCC_RECT:
        logger.trace(f"Removed  with H/W ration larger than: {hw_ratio}")
        valid_labels &= hwratio <= hw_ratio

    if method & FBCC_SIZE == FBCC_SIZE:
        logger.trace(f"Removed  with size larger than: {max_size}")
        valid_labels &= size <= max_size

    if method & FBCC_AREA == FBCC_AREA:
        if not min_area is None:
            valid_labels &= (area >= min_area) 
        if not max_area is None:
            valid_labels &= (area <= max_area)


    remaining_labels = np.where(valid_labels)[0] + 1
    removed_labels = np.where(~valid_labels)[0] + 1

    # Remove invalid connected components
    for label in tqdm(remaining_labels, desc="Removing Filtered Connected Components"):
        fcc_mask[labels_im == label] = 0

    logger.debug(f"Filtering (Removing) Connected Components removed {num_labels - len(remaining_labels) - 1} connected components")
    logger.trace(f"Removed labels: {removed_labels}")

    loss = (len(remaining_labels)  / num_labels) * 100
    removed_stats = stats[removed_labels]
    
    return fcc_mask, loss

def usingBlackInk(image: np.array, threshold: int = 50) -> float:
    """
    Calculate the percentage of black or near-black pixels in an image.

    Parameters:
    image (np.array): Input image, can be grayscale or color.
    threshold (int): Pixel value below which pixels are considered black or near-black.

    Returns:
    float: Percentage of black or near-black pixels in the image.
    """
    # Convert to grayscale if the image is not already grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the total number of pixels
    total_pixels = image.size

    # Calculate the number of black or near-black pixels
    black_pixels = np.sum(image <= threshold)

    # Calculate the percentage
    black_pixel_percentage = (black_pixels / total_pixels) 



    return black_pixel_percentage

def find_text_contours(binary_image):
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def extract_contours(image):
    """
    Extracts contours from the given image, filters out bounding contours, and contours inside other contours.

    Parameters:
    image (np.ndarray): Input image from which contours are to be extracted.

    Returns:
    extracted_contours (list of np.ndarray): List of regions of interest (ROIs) containing extracted contours.
    bb_image (np.ndarray): Image with bounding boxes drawn around the extracted contours.
    """
    
    # Convert to grayscale if the image is not already in grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find contours in the image
    contours, _ = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    logger.trace(f"Found {len(contours)} contours.")
    
    # Sort contours by the x value of the bounding rectangle
    contours = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[0])

    # Convert the image back to BGR for visualization
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image_height, image_width = image.shape[:2]
    
    extracted_contours = []
    bounding_boxes = [cv2.boundingRect(c) for c in contours] 

    filtered_bounding_boxes = []
    filtered_contours = []

    # Filter out bounding contours of the entire image
    for i, bb in enumerate(bounding_boxes):
        x, y, w, h = bb
        if w < image_width * 0.95 or h < image_height * 0.95:
            filtered_bounding_boxes.append(bb)
            filtered_contours.append(contours[i])
        else:
            logger.debug("Contour is the bounding contour of the entire image and will be removed")

    # Update the original lists with the filtered results
    bounding_boxes = filtered_bounding_boxes
    contours = filtered_contours

    # Copy the image for drawing bounding boxes
    bb_image = image_color.copy()
    
    # Iterate through the filtered contours
    for i, contour in enumerate(contours):
        x, y, w, h = bounding_boxes[i]
        
        # Check if the contour is inside another contour
        is_inner_contour = False
        for j, other_box in enumerate(bounding_boxes):
            if i != j:
                ox, oy, ow, oh = other_box
                if x > ox and y > oy and x + w < ox + ow and y + h < oy + oh:
                    is_inner_contour = True
                    break
        
        if not is_inner_contour:
            logger.debug(f"Contour {i}: x={x}, y={y}, w={w}, h={h}")
            # Draw the bounding box on the image
            cv2.rectangle(bb_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Extract the region of interest (ROI)
            roi = image_color[y:y+h, x:x+w]
            extracted_contours.append(roi)

    return extracted_contours, bb_image

def resize_and_pad(image, target_height=64, target_width=256):
    # Get original dimensions
    original_height, original_width = image.shape[:2]

    # Calculate aspect ratio
    aspect_ratio = original_width / original_height

    # Calculate new dimensions while maintaining aspect ratio
    if aspect_ratio > (target_width / target_height):
        # Width is the constraining dimension
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        # Height is the constraining dimension
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    # Create a new image of the target size with a black background
    padded_image = np.zeros((target_height, target_width), dtype=np.uint8)

    # Calculate padding offsets
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # Copy the resized image into the center of the padded image
    padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    return padded_image

def postProcessSegment(image: np.array,whiteThreshold = 99) -> np.array:
    height, width = image.shape[:2]
    image = crop_image_to_content(image)
    pct_white = percentWhite(image)
    if pct_white < whiteThreshold and (height > 20 and width > 20):
        image = np.bitwise_not(image)
        #image = resize_and_pad(image)
        image[image != 0] = 255
        return image
    return None

def deskew(img):
    """
    Deskews the input image by detecting and correcting the skew angle.

    Parameters:
    img (np.ndarray): Input grayscale image to be deskewed.

    Returns:
    np.ndarray: Deskewed image.
    """
    
    logger.info(f"Deskewing Image {img.shape}")
    
    # Add a black border around the image
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    # Detect edges using the Canny edge detector
    edges = cv2.Canny(img, 50, 200, apertureSize=3)
    
    # Define a kernel for morphological operations
    kernel = np.ones((3, 3), np.uint8)
    
    # Apply dilation and erosion to the image
    dilated = cv2.dilate(edges, kernel, iterations=1)
    dilated = cv2.erode(dilated, kernel, iterations=1)

    # Detect lines using the Hough Line Transform
    lines = cv2.HoughLines(dilated, 10, np.pi/1000, 10)
    tlines = []
    line_image = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    angles = []

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            delta_x = x2 - x1
            delta_y = y2 - y1
            
            # Calculate angle in radians
            angle_rad = math.atan2(delta_y, delta_x)
            
            # Convert angle to degrees
            angle_deg = math.degrees(angle_rad)
            
            # Filter for angles within a specific range
            if -43 < angle_deg < -40:
                logger.trace(f"Detected line angle: {angle_deg}")
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                tlines.append(line)
                angles.append(angle_deg)
    
    logger.info(f"Candidate angles: {angles}")

    # If no valid angles are found, return the original image
    if len(angles) == 0:
        return img  
    
    # Calculate the average angle for deskewing
    angle_avg = abs(sum(angles) / len(angles))
    logger.info(f"Average Deskew Angle: {angle_avg}")

    angle_rad = np.deg2rad(angle_avg)
    
    # Get the image dimensions
    (h, w) = img.shape[:2]
    
    # Calculate the skew transformation matrix
    skew_matrix = np.array([
        [1, np.tan(angle_rad), 0],
        [0, 1, 0]
    ], dtype=np.float32)
    
    # Apply the affine transformation to deskew the image
    deskewed_image = cv2.warpAffine(img, skew_matrix, (w + 50, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return 255 - crop_image_to_content(255 - deskewed_image)