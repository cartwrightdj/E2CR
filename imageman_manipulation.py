import cv2
import numpy as np
from common import *
from scipy.stats import kurtosis, skew, mode
from skimage.measure import shannon_entropy
from tqdm import tqdm
from loguru import logger
from PIL import Image


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

def preProcessImage(baseImage,
                    applyDenoise=PreProcessing.APPLY_DENOIS, 
                    useAdaptiveThreshold=PreProcessing.APPLY_ADAPTIVE_THRESHOLD, 
                    applyErode=PreProcessing.APPLY_ERODE,
                    erodeKernel=PreProcessing.erodeKernel,
                    threshold=PreProcessing.threshold,
                    applyDilation=PreProcessing.APPLY_DILATION,
                    dilateKernalSize=PreProcessing.DILATE_KERNEL,
                    applyMorphology=PreProcessing.APPLY_MORPHOLOGY) -> np.ndarray:
    """
    Preprocess an image with optional denoising, adaptive thresholding, erosion, dilation, and morphology.

    Arguments:
    baseImage -- np.ndarray, the input image
    applyDenoise -- bool, whether to apply denoising
    useAdaptiveThreshold -- bool, whether to use adaptive thresholding
    applyErode -- bool, whether to apply erosion
    erodeKernel -- np.ndarray, the kernel to use for erosion
    threshold -- int, the threshold value for binary thresholding
    applyDilation -- bool, whether to apply dilation
    dilateKernalSize -- tuple, the kernel size for dilation
    applyMorphology -- bool, whether to apply morphological operations

    Returns:
    ppImage -- np.ndarray, the preprocessed image
    """
    logger.info("Starting image preprocessing")
    logger.debug(print(f"applyDenoise={applyDenoise}, useAdaptiveThreshold={useAdaptiveThreshold}, applyErode={applyErode}, erodeKernel={erodeKernel}\n\t Threshold={threshold},applyDilation={applyDilation}, dilateKernalSize={dilateKernalSize}, applyMorphology={applyMorphology}"))
    
    step = 0
    try:
        # Convert to grayscale if the image is not already grayscale
        if len(baseImage.shape) != 2:
            logger.info("Converting image to grayscale")
            ppImage = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
        else:
            ppImage = baseImage
    except cv2.error as E:
        logger.critical(f"Pre-Processing received no image: {E.msg}")
        return baseImage

    # Save the intermediate result
    #gray_image_path = os.path.join(os.getcwd(),"output", "base_to_gray.png")
    #cv2.imwrite(gray_image_path, ppImage)
    #logger.debug(f"Saved grayscale image to {gray_image_path}")
    

    #Detect and Remove Border
    step += 1
    ppImage = remove_border(baseImage)
    cv2.imwrite(os.path.join(os.getcwd(),"output", f"Step_{step}_Border_Removed.png"), ppImage)
    
  
    # Apply denoising
    if applyDenoise:
        step += 1
        logger.info("Applying denoising")
        ppImage = cv2.fastNlMeansDenoising(ppImage, None, 19, 5, 5)
        
        denoise_image_path = os.path.join(os.getcwd(),"output", f"Step_{step}_Denoised_Image.png")
        cv2.imwrite(denoise_image_path, ppImage)
        logger.debug(f"Saved denoised image to {denoise_image_path}")
    
    # Apply erosion
    if applyErode:
        step += 1
        logger.info("Applying erosion")
        ppImage = cv2.convertScaleAbs(ppImage, alpha=1, beta=0)
        ppImage = cv2.erode(ppImage, erodeKernel)
        erode_image_path = os.path.join(os.getcwd(),"output", f"Step_{step}_Eroded_Image.png")
        cv2.imwrite(erode_image_path, ppImage)
        logger.debug(f"Saved eroded image to {erode_image_path}")

    # Apply adaptive thresholding or simple thresholding
    if useAdaptiveThreshold:
        step += 1
        logger.info("Using adaptive thresholding")
        ppImage = cv2.adaptiveThreshold(ppImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 4)
        ppImage = cv2.fastNlMeansDenoising(ppImage, None, 19, 5, 5)
        adaptive_thresh_image_path = os.path.join(os.getcwd(),"output", f"Step_{step}_Image__AdaptiveThreshold.png")
        cv2.imwrite(adaptive_thresh_image_path, ppImage)
        logger.debug(f"Saved adaptively thresholded image to {adaptive_thresh_image_path}")
    else:
        logger.info("Using simple thresholding")
        _, ppImage = cv2.threshold(ppImage, threshold, 255, cv2.THRESH_BINARY)
        thresh_image_path = os.path.join(os.getcwd(),"output", f"Step_{step}_Image__AdaptiveThreshold.png")
        cv2.imwrite(thresh_image_path, ppImage)
        logger.debug(f"Saved thresholded image to {thresh_image_path}")

    step += 1
    ppImage = remove_small_black_areas(ppImage, 95)
    spotRem_image_path = os.path.join(os.getcwd(),"output", f"Step_{step}_Image_SpotRemoval.jpg")
    cv2.imwrite(spotRem_image_path, ppImage)
    logger.debug(f"Saved thresholded image to {spotRem_image_path}")

    # Apply dilation
    if applyDilation:
        step += 1
        logger.info("Applying dilation")
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilateKernalSize)
        ppImage = cv2.dilate(ppImage, rect_kernel, iterations=1)
        dilation_image_path = os.path.join(os.getcwd(),"output", f"Step_{step}_Dilated_Image.png")
        cv2.imwrite(dilation_image_path, ppImage)
        logger.debug(f"Saved dilated image to {dilation_image_path}")

    # Apply morphological operations
    if applyMorphology:
        step += 1
        logger.info("Applying morphological operations")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        ppImage = cv2.morphologyEx(ppImage, cv2.MORPH_CLOSE, kernel)
        morphology_image_path = os.path.join(os.getcwd(),"output", f"Step_{step}_Image_Morphology.png")
        cv2.imwrite(morphology_image_path, ppImage)
        logger.debug(f"Saved morphologically processed image to {morphology_image_path}")
    
    logger.info("Finished image preprocessing")
    return ppImage

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

def draw_path_on_image(image, path, max_y, min_y):
    """
    Draw the shortest path on the image.

    Arguments:
    image -- np.ndarray, the original image
    path -- list of tuples, the sequence of points in the shortest path
    max_y -- int, the maximum y-coordinate in the path
    min_y -- int, the minimum y-coordinate in the path

    Returns:
    output_image -- np.ndarray, the image with the path drawn
    """
    if len(image.shape) == 2:  # If the image is grayscale
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        output_image = image     
    
    for (cy, cx) in path:
        output_image[cy, cx] = [255, 0, 0]

    # Annotate max and min y-values
    height, width = output_image.shape[:2]
    if path:
        cv2.putText(output_image, f"Max Y: {max_y}", (int((width /2 )-5), max_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
        cv2.putText(output_image, f"Min Y: {min_y}", (int((width / 2)-5), min_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

    return output_image

