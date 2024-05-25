import cv2
import numpy as np
from common import *
from scipy.stats import kurtosis, skew, mode
from skimage.measure import shannon_entropy
from tqdm import tqdm
from loguru import logger
from PIL import Image

def crop_image(image, values, axis, min_size=DefaultSeg.EXPECTED_IMAGE_HW):    
    """
    Crops the image based on the given list of values along the specified axis,
    ensuring no cropped section is smaller than the specified threshold and
    combining skipped crops with the smaller of the adjacent sections.
    
    Args:
        image (np.array): The image to crop.
        values (list): List of values where the image will be cropped.
        axis (int): Axis along which to crop (0 for y-axis, 1 for x-axis).
        min_size (int): Minimum size for any cropped section.
    
    Returns:
        tuple: A tuple containing the list of cropped image sections and the number of skipped crops.
    """
    logger.trace(f"Cropping image by {len(values)} crops, acis={axis}, min_size={min_size}")

    cropped_images = []
    skipped_crops = 0
    start = 0
    height, width = image.shape[:2]

    if axis == 0:
        max_limit = height
    elif axis == 1:
        max_limit = width
    else:
        raise ValueError("Axis must be 0 (y-axis) or 1 (x-axis)")

    for i, value in enumerate(values):
        if start < value < max_limit:
            if (value - start) < min_size:
                skipped_crops += 1
                # Decide whether to merge with the previous or next section
                if cropped_images:
                    prev_size = value - start + cropped_images[-1].shape[axis]
                else:
                    prev_size = max_limit + 1  # Infinity equivalent to always choose next

                next_size = (values[i + 1] - start) if (i + 1 < len(values) and values[i + 1] < max_limit) else (max_limit - start)
                
                if prev_size < next_size and cropped_images:
                    if axis == 0:
                        cropped_images[-1] = np.vstack((cropped_images[-1], image[start:value, :]))
                    else:
                        cropped_images[-1] = np.hstack((cropped_images[-1], image[:, start:value]))
                else:
                    continue  # Skip small crop and handle it in the next iteration
            else:
                if axis == 0:
                    cropped_images.append(image[start:value, :])
                else:
                    cropped_images.append(image[:, start:value])
            start = value

    # Handle the last section
    if start < max_limit:
        if (max_limit - start) >= min_size:
            if axis == 0:
                cropped_images.append(image[start:height, :])
            else:
                cropped_images.append(image[:, start:width])
        elif cropped_images:
            skipped_crops += 1
            # Combine the last section with the last cropped image
            if axis == 0:
                cropped_images[-1] = np.vstack((cropped_images[-1], image[start:height, :]))
            else:
                cropped_images[-1] = np.hstack((cropped_images[-1], image[:, start:width]))
        else:
            # Handle case where no crop was made yet
            if axis == 0:
                cropped_images.append(image[start:height, :])
            else:
                cropped_images.append(image[:, start:width])
    
    if skipped_crops != 0:
        logger.trace(f"E2CR Image Cropping skipped. Image is being returned, height was {height}, width was {width}")
        
    logger.trace(f"E2CR Image Cropped, returning  {len(cropped_images)} crops, acis={axis}, min_size={min_size}")
    return cropped_images, skipped_crops

def write_text_on_image(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale=1, color=(255, 0, 0), thickness=2):
    """
    Write text on an image using OpenCV.
    
    Args:
        image (np.array): The image on which to write the text.
        text (str): The text to write.
        position (tuple): The position (x, y) at which to write the text.
        font (int): Font type. Default is cv2.FONT_HERSHEY_SIMPLEX.
        font_scale (float): Font scale factor that is multiplied by the font-specific base size.
        color (tuple): Text color in BGR. Default is blue (255, 0, 0).
        thickness (int): Thickness of the text stroke. Default is 2.
    
    Returns:
        np.array: The image with the text written on it.
    """
    cv2.putText(image, text, position, font, font_scale, color, thickness)
    return image

def draw_horizontal_lines(image, y_values, color=(0, 255, 0), thickness=2,label=True):
    """
    Draws horizontal lines on the given image at the specified y-values.

    Args:
        image (np.array): The image on which to draw lines.
        y_values (list): List of y-values where the lines will be drawn.
        color (tuple): Color of the lines in (B, G, R) format. Default is green.
        thickness (int): Thickness of the lines. Default is 2.

    Returns:
        np.array: The image with the lines drawn.
    """
    # Make a copy of the image to draw on
    
    image_with_lines = image.copy()

    if len(image_with_lines.shape) == 2 or (len(image_with_lines) == 3 and image_with_lines.shape[2] == 1):
        # Convert to 3-channel BGR image
        image_with_lines = cv2.cvtColor(image_with_lines, cv2.COLOR_GRAY2BGR)
    
    width = image.shape[1]
    
    for y in y_values:
        
        cv2.line(image_with_lines, (0, y), (width, y), color, thickness)
        if label:
            image_with_lines = write_text_on_image(image_with_lines,str(y),(5, y-3))  
           
    return image_with_lines

def draw_vertical_lines(image, x_values, color=(0, 255, 0), thickness=2):
    """
    Draws horizontal lines on the given image at the specified y-values.

    Args:
        image (np.array): The image on which to draw lines.
        y_values (list): List of y-values where the lines will be drawn.
        color (tuple): Color of the lines in (B, G, R) format. Default is green.
        thickness (int): Thickness of the lines. Default is 2.

    Returns:
        np.array: The image with the lines drawn.
    """
    # Make a copy of the image to draw on
    image_with_lines = image.copy()
    
    height = image.shape[0]
    
    for x in x_values:
        cv2.line(image_with_lines, (x, 0), (x, height), color, thickness)
    
    return image_with_lines

def convert_to_black_and_white(image, threshold=150):
    """
    Convert a cv2 image to black and white (binary image).
    
    Args:
        image (np.array): The input image.
        threshold (int): The threshold value used to convert the grayscale image to binary. Default is 127.
    
    Returns:
        np.array: The black and white image.
    """
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply binary thresholding
    _, bw_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

    bw_image = cv2.fastNlMeansDenoising(bw_image, None, 19, 21, 21 )
    
    return bw_image

def get_mode(data):
    """
    Calculate the mode of the given data.
    
    Args:
        data (np.array): The input data.
    
    Returns:
        int or None: The mode of the data or None if it cannot be calculated.
    """
    mode_result = mode(data, axis=None)
    if mode_result.count.size > 0:
        return mode_result.mode
    else:
        return None

def get_image_statistics(image):
    """
    Calculate and return statistics for an image.
    
    Args:
        image (np.array): The input image.
    
    Returns:
        dict: A dictionary containing image statistics.
    """
    stats = {}
    if len(image.shape) == 2:
        # Grayscale image
        stats['mean'] = np.mean(image)
        stats['std_dev'] = np.std(image)
        stats['min'] = np.min(image)
        stats['max'] = np.max(image)
        stats['median'] = np.median(image)
        stats['mode'] = get_mode(image)
        stats['skewness'] = skew(image, axis=None)
        stats['kurtosis'] = kurtosis(image, axis=None)
        stats['entropy'] = shannon_entropy(image)
        stats['histogram'] = cv2.calcHist([image], [0], None, [256], [0, 256])
    else:
        # Color image
        channels = cv2.split(image)
        stats['mean'] = [np.mean(ch) for ch in channels]
        stats['std_dev'] = [np.std(ch) for ch in channels]
        stats['min'] = [np.min(ch) for ch in channels]
        stats['max'] = [np.max(ch) for ch in channels]
        stats['median'] = [np.median(ch) for ch in channels]
        stats['mode'] = [get_mode(ch) for ch in channels]
        stats['skewness'] = [skew(ch, axis=None) for ch in channels]
        stats['kurtosis'] = [kurtosis(ch, axis=None) for ch in channels]
        stats['entropy'] = [shannon_entropy(ch) for ch in channels]
        stats['histogram'] = [cv2.calcHist([ch], [0], None, [256], [0, 256]) for ch in channels]
    
    return stats
  
def print_image_statistics(image):
    stats = get_image_statistics(image)
    is_color=is_color_image(image)
    """
    Print the statistics of an image in a readable format.
    
    Args:
        stats (dict): A dictionary containing image statistics.
        is_color (bool): Flag indicating if the image is color or grayscale.
    """
    if not is_color:
        # Grayscale image
        print("Image Statistics (Grayscale):")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Standard Deviation: {stats['std_dev']:.2f}")
        print(f"  Min: {stats['min']}")
        print(f"  Max: {stats['max']}")
        print(f"  Median: {stats['median']}")
        print(f"  Mode: {stats['mode']}")
        print(f"  Skewness: {stats['skewness']:.2f}")
        print(f"  Kurtosis: {stats['kurtosis']:.2f}")
        print(f"  Entropy: {stats['entropy']:.2f}")
        #print(f"  Histogram: {stats['histogram'].flatten().tolist()}")
    else:
        # Color image
        channels = ['Blue', 'Green', 'Red']
        print("Image Statistics (Color):")
        for i, channel in enumerate(channels):
            if channel == "Blue": print(Colors.OKBLUE)
            if channel == "Green": print(Colors.OKGREEN)
            if channel == "Red": print(Colors.OKRED)
            print(f"  {channel} Channel:")
            print(f"    Mean: {stats['mean'][i]:.2f}")
            print(f"    Standard Deviation: {stats['std_dev'][i]:.2f}")
            print(f"    Min: {stats['min'][i]}")
            print(f"    Max: {stats['max'][i]}")
            print(f"    Median: {stats['median'][i]}")
            print(f"    Mode: {stats['mode'][i]}")
            print(f"    Skewness: {stats['skewness'][i]:.2f}")
            print(f"    Kurtosis: {stats['kurtosis'][i]:.2f}")
            print(f"    Entropy: {stats['entropy'][i]:.2f}")
            #print(f"    Histogram: {stats['histogram'][i].flatten().tolist()}")
            print(Colors.ENDC)

def is_color_image(image):
    """
    Check if an image is a color image.

    Args:
        image (np.array): The input image.

    Returns:
        bool: True if the image is color, False if it is grayscale.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        return True
    elif len(image.shape) == 2:
        return False
    else:
        raise ValueError("Input image must be either a 2D grayscale image or a 3D color image")


def find_and_fill_edges(image):
    """
    Find edges in an image and fill them in.
    
    Args:
        image (np.array): The input image.
    
    Returns:
        np.array: The image with edges filled in.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Invert the edges image
    edges_inv = cv2.bitwise_not(edges)
    
    # Perform morphological operations to fill in the edges
    kernel = np.ones((5, 5), np.uint8)
    filled_edges = cv2.morphologyEx(edges_inv, cv2.MORPH_CLOSE, kernel)
    
    # Convert the filled edges to 3-channel image
    filled_edges_color = cv2.cvtColor(filled_edges, cv2.COLOR_GRAY2BGR)
    
    return filled_edges_color

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

def detect_lines(image, rho=1, theta=np.pi/180, threshold=15, min_line_length=30, max_line_gap=10, contour_area_threshold=100):
    """
    Detect isolated lines in an image, excluding lines that are part of complex shapes.

    Args:
        image (np.array): The input image.
        rho (float): Distance resolution of the accumulator in pixels.
        theta (float): Angle resolution of the accumulator in radians.
        threshold (int): Accumulator threshold parameter. Only those lines are returned that get enough votes.
        min_line_length (int): Minimum line length. Line segments shorter than this are rejected.
        max_line_gap (int): Maximum allowed gap between points on the same line to link them.
        contour_area_threshold (int): Minimum area of contours to be considered as part of complex shapes.

    Returns:
        np.array: The image with detected isolated lines drawn.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy.ndarray")
    
    # Convert to grayscale if the input is a color image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Edge detection using Canny
    edges = cv2.Canny(gray, 50, 200)
    
    # Find contours to detect complex shapes
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for complex shapes
    mask = np.zeros_like(edges)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > contour_area_threshold:
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Find lines using the Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    # Convert grayscale image to BGR to draw colored lines
    img_dst = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Draw found lines that are not part of complex shapes
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if mask[y1, x1] == 0 and mask[y2, x2] == 0:  # Check if the line endpoints are not in the mask
                isolated = True
                for contour in contours:
                    if cv2.pointPolygonTest(contour, (int(x1), int(y1)), False) >= 0 or cv2.pointPolygonTest(contour, (int(x2), int(y2)), False) >= 0:
                        isolated = False
                        break
                if isolated:
                    cv2.line(img_dst, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img_dst

def remove_lines(image):
    """
    Locates and removes vertical and horizontal lines from an image.
    """
    # Convert the image to grayscale
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding to create a binary image
    gray_image = image
    
    binary_image = cv2.adaptiveThreshold(~gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    # Create horizontal and vertical kernels
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

    # Detect horizontal lines
    horizontal_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # Detect vertical lines
    vertical_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Combine detected lines
    lines = cv2.add(horizontal_lines, vertical_lines)

    # Invert the lines image
    inverted_lines = cv2.bitwise_not(lines)

    # Remove lines from the original image
    final_image = cv2.bitwise_and(gray_image, gray_image, mask=inverted_lines)

    # Convert the grayscale image back to BGR for consistency
    final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_GRAY2BGR)

    return final_image_bgr

def ensure_color(image):
    """
    Ensure the image is in BGR color format.
    
    Args:
        image (np.array): The input image.

    Returns:
        np.array: The image in BGR color format.
    """
    if len(image.shape) == 2:  # If the image is grayscale
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image

def combine_images(image_grid, label=True):
    """
    Combine a two-dimensional list of images into one large image with rectangles around each image.
    Optionally label each box with the row and column ID.

    Args:
        image_grid (list of list of np.array): 2D list of images to combine.
        label (bool): Whether to label each box with the row and column ID.

    Returns:
        tuple: (combined_image, coordinates)
            combined_image (np.array): The combined image.
            coordinates (list of tuple): List of ((x1, y1), (x2, y2), row_idx, col_idx) coordinates for the boxes.
    """
    # Check if the grid is not empty
    if not image_grid or not image_grid[0]:
        raise ValueError("Image grid must not be empty")

    # Convert all images to BGR color format
    for row in image_grid:
        for i in range(len(row)):
            row[i] = ensure_color(row[i])

    # Combine images row by row
    combined_rows = []
    for row in image_grid:
        combined_row = cv2.hconcat(row)
        combined_rows.append(combined_row)

    # Stack all combined rows vertically
    combined_image = cv2.vconcat(combined_rows)

    # Draw rectangles around each image and optionally label them
    coordinates = []  # List to store the coordinates of each box
    y_offset = 0
    for row_idx, row in enumerate(image_grid):
        x_offset = 0
        for col_idx, img in enumerate(row):
            img_height, img_width = img.shape[:2]
            top_left = (x_offset, y_offset)
            bottom_right = (x_offset + img_width, y_offset + img_height)
            cv2.rectangle(combined_image, top_left, bottom_right, (0, 255, 0), 2)
            if label:
                text = f"({row_idx}, {col_idx})"
                cv2.putText(combined_image, text, (x_offset + 5, y_offset + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Add the coordinates of the top-left and bottom-right corners along with row and column indices
            coordinates.append((top_left, bottom_right, row_idx, col_idx))
            x_offset += img_width
        y_offset += img_height

    return combined_image, coordinates

def fill_enclosed_areas(image):
    """
    Fill enclosed areas in an image.

    Args:
        image (np.array): The input image.

    Returns:
        np.array: The image with enclosed areas filled.
    """
    # Convert to grayscale if the input is a color image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Threshold the image to get a binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Invert the binary image
    binary = cv2.bitwise_not(binary)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for flood filling
    h, w = binary.shape
    mask = np.zeros((h+2, w+2), np.uint8)

    # Fill each contour
    for contour in contours:
        cv2.drawContours(mask, [contour], 0, 255, -1)  # Draw filled contour on mask
        cv2.floodFill(binary, mask, (0, 0), 255)  # Flood fill from the top-left corner

    # Invert the filled binary image
    filled_image = cv2.bitwise_not(binary)

    # Convert filled image to BGR if the input was color
    if len(image.shape) == 3:
        filled_image = cv2.cvtColor(filled_image, cv2.COLOR_GRAY2BGR)

    return filled_image

def remove_black_border_using_histogram(image):
    """
    Remove irregular black border from an image using histograms of pixel sums.

    Args:
        image (np.array): The input image.

    Returns:
        np.array: The image with the black border removed, retaining the same color channels.
    """
    # Ensure the image is not empty
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or invalid")

    # Convert to grayscale if the input is a color image
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    # Calculate the sum of pixels for each row and column
    row_sums = np.sum(image, axis=1)
    col_sums = np.sum(image, axis=0)

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

    # Crop the image using the bounding box
    cropped_image = image[y_min:y_max+1, x_min:x_max+1]

    return cropped_image

def preProcessImage(ppImage):
    #print(f"{Colors.OKBLUE}PreProcessing Image{Colors.ENDC}")
    pbar = tqdm(total=4,desc=f"{Colors.OKBLUE}PreProcessing Image{Colors.ENDC}")
    cv2.imwrite(f"e:/E2CR/preProcess/origional.jpg", ppImage)
    

    pbar.desc=(f"{Colors.OKGREEN}Removing any dark borders{Colors.ENDC}")
    ppImage = remove_black_border_using_histogram(ppImage)
    cv2.imwrite(f"e:/E2CR/preProcess/remove_black_border_using_histogram.jpg", ppImage)
    pbar.update(1)
    

    pbar.desc=(f"{Colors.OKGREEN}Converting Image to Graycale{Colors.ENDC}")
    # Convert to grayscale if the input is a color image
    if len(ppImage.shape) == 3:
        ppImage = cv2.cvtColor(ppImage, cv2.COLOR_BGR2GRAY)
    pbar.update(1)

    pbar.desc=(f"{Colors.OKGREEN}Removing Noise{Colors.ENDC}")
    ppImage = cv2.fastNlMeansDenoising(ppImage, None, 10, 7, 21 )
    cv2.imwrite(f"e:/E2CR/preProcess/fastNlMeansDenoising.jpg", ppImage)
    pbar.update(1)

    pbar.desc=(f"{Colors.OKGREEN}Conducting Adaptive Threshold{Colors.ENDC}")
    ppImage = cv2.adaptiveThreshold(ppImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 4)
    cv2.imwrite(f"e:/E2CR/preProcess/adaptiveThreshold.jpg", ppImage)
    pbar.update(1)
    pbar.desc=(f"{Colors.OKGREEN}Image Pre-Processing Complete{Colors.ENDC}")
    
    #_, ppImage = cv2.threshold(ppImage, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #ppImage = remove_small_black_areas(ppImage,95)
    
    cv2.imwrite(f"E:/OCR/lines/thresh_.jpg", ppImage)

   

    return ppImage

def find_first_non_white_column(image):
    """
    Finds the first column from the left that is not completely white.
    """
    for col in range(image.shape[1]):
        if not np.all(image[:, col] == 255):  # Check if the entire column is not white
            return col
    return None

def make_boxes(image, width):
    """
    Creates boxes of the specified width starting from the first non-white column.
    """
    
    first_non_white_col = find_first_non_white_column(image)
    
    if first_non_white_col is None:
        print("The image is completely white.")
        

    boxes = []
    for col in range(first_non_white_col, image.shape[1], width):
        box = image[:, col:col+width]
        boxes.append(box)
        # Draw the box on the original image
        cv2.rectangle(image, (col, 0), (col + width, image.shape[0]), (100, 125, 255), 2)
    
    #return boxes, image
    
    return image

def draw_boxes(image, coordinates):
    """
    Draws boxes and labels on the given image using the list of coordinates.

    Args:
        image (np.array): The image on which to draw the boxes.
        coordinates (list of tuple): List of ((x1, y1), (x2, y2), row_idx, col_idx) coordinates for the boxes.

    Returns:
        np.array: The image with boxes and labels drawn.
    """
    # Ensure the image is in BGR color format
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image

    for (top_left, bottom_right, row_idx, col_idx) in coordinates:
        x1, y1 = top_left
        x2, y2 = bottom_right
        
        # Draw rectangle on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Label the box with row and column indices
        label = f"({row_idx}, {col_idx})"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw the beige background rectangle for the text
        cv2.putText(image, label, (x1+5, y1 +20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        

        
        # Put the text on top of the beige rectangle
        

    return image

def crop_to_non_white_areas(image):
    """
    Crops the image to include only non-white areas.
    """
    # Convert the image to grayscale
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find the non-white mask
    gray_image = remove_lines(image)
    non_white_mask = gray_image != 255
    
    # Find the bounding box of the non-white areas
    coords = np.column_stack(np.where(non_white_mask))
    if len(coords) == 0:
        return image  # Return the original image if it's completely white

    top_left = coords.min(axis=0)
    bottom_right = coords.max(axis=0)

    # Crop the image
    cropped_image = image[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
    return cropped_image

def detect_lines(image, canny_threshold1=50, canny_threshold2=150, canny_apertureSize=3, 
                 hough_rho=1, hough_theta=np.pi / 180, hough_threshold=150):
    """
    Detect lines in the given image and return a list of lines.
    Each line is represented by its parameters (rho, theta).

    Parameters:
    - image: Input image
    - canny_threshold1: First threshold for the hysteresis procedure in Canny edge detector
    - canny_threshold2: Second threshold for the hysteresis procedure in Canny edge detector
    - canny_apertureSize: Aperture size for the Sobel operator in Canny edge detector
    - hough_rho: Distance resolution of the accumulator in pixels in Hough Line Transform
    - hough_theta: Angle resolution of the accumulator in radians in Hough Line Transform
    - hough_threshold: Accumulator threshold parameter in Hough Line Transform
    """
    def detect_lines_in_image(gray_image):
        """
        Detect lines using Hough Line Transform.
        """
        edges = cv2.Canny(gray_image, canny_threshold1, canny_threshold2, apertureSize=canny_apertureSize)
        lines = cv2.HoughLines(edges, hough_rho, hough_theta, hough_threshold)
        return lines

    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Detect lines in the image
    lines = detect_lines_in_image(gray_image)

    # Convert the lines to a list of tuples (rho, theta)
    line_list = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            line_list.append((rho, theta))

    return line_list

def display_lines(image, lines):
    """
    Display the image with detected lines overlaid.
    """
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    print(f"here i am {len(lines)}, {lines}")
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(color_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    return color_image

def correct_image_tilt(image):
    """
    Recognize tilt in an image and return the untilted image.
    """
    def detect_tilt_angle(gray_image):
        """
        Detect the tilt angle of the image using Hough Line Transform.
        """
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

        if lines is not None:
            meanAngle = 0
            # Set min number of valid lines (try higher)
            numLines = np.sum(1 for l in lines if l[0][1] < 0.7 or l[0][1] > 2.6)
            if numLines > 1:
                meanAngle = np.mean([l[0][1] for l in lines if l[0][1] < 0.7 or l[0][1] > 2.6])

                # Look for angle with correct value
                if meanAngle != 0 and (meanAngle < 0.7 or meanAngle > 2.6):
                    return meanAngle
        
        return 0

    def rotate_image(image, angle):
        """
        Rotate the image by the given angle.
        """
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Detect the tilt angle
    tilt_angle = detect_tilt_angle(gray_image)

    # Rotate the image to correct the tilt
    untilted_image = rotate_image(image, -tilt_angle)

    return untilted_image

def set_image_dpi(file_path):

    im = Image.open(file_path)
    length_x, width_y = im.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    im_resized = im.resize(size)
    im_resized.save(file_path, dpi=(300, 300))

# Image PreProccesing
def preProcessImage(baseImage,
                    applyDenoise = PreProcessing.APPLY_DENOIS, 
                    useAdaptiveThreshold = PreProcessing.APPLY_ADAPTIVE_THRESHOLD, 
                    applyErode=PreProcessing.APPLY_ERODE,
                    erodeKernel = PreProcessing.erodeKernel,
                    threshold = PreProcessing.threshold,
                    applyDilation = PreProcessing.APPLY_DILATION,
                    dilateKernalSize = PreProcessing.DILATE_KERNEL) -> np.ndarray:
    
    ppImage = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(cwd,"base_to_gray.png"),ppImage)
    
    if applyDenoise:
        ppImage = cv2.fastNlMeansDenoising(ppImage, None, 19, 5, 5 )
        cv2.imwrite(f"E:/OCR/gray_denoise.png",ppImage)
    
    if applyErode:
        ppImage = cv2.convertScaleAbs(ppImage, alpha=1, beta=0)
        print("Eroding base Image")
        ppImage = cv2.erode(ppImage, erodeKernel)
        cv2.imwrite(os.path.join(cwd,"erode.png"),ppImage)


    if useAdaptiveThreshold:
        print("Using adaptive threshold")
        ppImage = cv2.adaptiveThreshold(ppImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 4)
        ppImage = cv2.fastNlMeansDenoising(ppImage, None, 19, 5, 5 )
    else:
        ret, ppImage = cv2.threshold(ppImage, threshold, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(cwd,"thresh.png"),ppImage)

    

    if applyDilation:
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilateKernalSize)
        ppImage = cv2.dilate(ppImage, rect_kernel, iterations = 1)
        cv2.imwrite(os.path.join(cwd,"dilation.png"),ppImage)
    
    return ppImage

def resize_image(image, target_size=(128, 32)):
    """
    Resize the image to the target size while maintaining the aspect ratio.
    Center the resized image based on the greater dimension.
    """
    target_w, target_h = target_size
    h, w = image.shape[:2]

    # Calculate the scaling factor to maintain aspect ratio
    scaling_factor = min(target_w / w, target_h / h)

    # Resize the image using the scaling factor
    new_w = int(w * scaling_factor)
    new_h = int(h * scaling_factor)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a blank image with the target size
    target_image = np.ones((target_h, target_w), dtype=np.uint8) * 255

    # Calculate the padding to center the resized image based on the greater axis
    result = target_image.copy()
    if new_w < target_w:
        # Width is smaller, center horizontally
        pad_x = (target_w - new_w) // 2
        pad_y = 0

    else:
        # Height is smaller, center vertically
        pad_x = 0
        pad_y = (target_h - new_h) // 2
    
    result[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_image
    h, w = result.shape[:2]
    if h != target_h or w != target_w:
        assert False
    
    

    return result
