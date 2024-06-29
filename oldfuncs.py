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

def remove_lines(image, 
                 canny_threshold1=50, canny_threshold2=150, 
                 gaussian_blur_ksize=(5, 5), gaussian_blur_sigma=1.5, 
                 hough_rho=1, hough_theta=np.pi / 180, hough_threshold=200, 
                 hough_min_line_length=50, hough_max_line_gap=10,
                 apply_morphology=False, morph_kernel_size=(5, 5), morph_iterations=1,
                 angle_threshold=np.pi / 175,
                 side_threshold=200):
    """
    Detects generally vertical and horizontal lines in an image using Canny edge detection and Hough Line Transform.
    Only considers lines within a specified threshold from the sides of the image.

    Parameters:
    - image (ndarray): Input image.
    - canny_threshold1 (int): First threshold for the hysteresis procedure in Canny.
    - canny_threshold2 (int): Second threshold for the hysteresis procedure in Canny.
    - gaussian_blur_ksize (tuple): Kernel size for Gaussian blur.
    - gaussian_blur_sigma (float): Standard deviation for Gaussian blur.
    - hough_rho (float): Distance resolution of the accumulator in pixels.
    - hough_theta (float): Angle resolution of the accumulator in radians.
    - hough_threshold (int): Accumulator threshold parameter for Hough Line Transform.
    - hough_min_line_length (int): Minimum length of a line to be detected by the Probabilistic Hough Line Transform.
    - hough_max_line_gap (int): Maximum allowed gap between points on the same line to link them by the Probabilistic Hough Line Transform.
    - apply_morphology (bool): If True, apply morphological transformations to the edge map.
    - morph_kernel_size (tuple): Kernel size for morphological operations.
    - morph_iterations (int): Number of iterations for morphological operations.
    - angle_threshold (float): Threshold to filter lines based on angle.
    - side_threshold (int): Pixel threshold from the sides of the image to consider for line detection.

    Returns:
    - output_img (ndarray): Image with detected lines drawn.
    """
    
    if image is None:
        raise ValueError("Image not found or path is incorrect")
    
    # Convert to grayscale if the image is in color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, gaussian_blur_ksize, gaussian_blur_sigma)
    
    # Detect edges using Canny
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)
    
    # Apply morphological transformations if required
    if apply_morphology:
        kernel = np.ones(morph_kernel_size, np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=morph_iterations)
        edges = cv2.erode(edges, kernel, iterations=morph_iterations)
    
    # Apply Hough Line Transform
    lines = cv2.HoughLines(edges, hough_rho, hough_theta, hough_threshold)
    
    # Get image dimensions
    height, width = image.shape[:2]

    # Draw the vertical and horizontal lines detected by Hough Line Transform
    if len(image.shape) == 2:
        image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
       
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            # Filter based on angle
            if (abs(theta) < angle_threshold or
                abs(theta - np.pi/2) < angle_threshold or
                abs(theta - np.pi) < angle_threshold or
                abs(theta - 3*np.pi/2) < angle_threshold):
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Apply Probabilistic Hough Line Transform
    linesP = cv2.HoughLinesP(edges, hough_rho, hough_theta, hough_threshold, 
                             minLineLength=hough_min_line_length, maxLineGap=hough_max_line_gap)
    
    # Draw the vertical and horizontal lines detected by Probabilistic Hough Line Transform
    if linesP is not None:
        for line in linesP:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            # Filter based on angle
            if (abs(angle) < angle_threshold or
                abs(angle - np.pi/2) < angle_threshold or
                abs(angle - np.pi) < angle_threshold or
                abs(angle - 3*np.pi/2) < angle_threshold):
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Create a mask for the area outside the side_threshold
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[side_threshold:-side_threshold, side_threshold:-side_threshold] = 255

    # Apply the mask to the image with drawn lines
    masked_image = cv2.bitwise_and(image, mask)
    masked_image[mask == 0] = 255

    return masked_image

def cumulative_sum(image, subOnDecrease: bool = DefaultParameters.subOnDecrease):
    """
    Combines the functionalities of summing pixel values along x and y axes, and creating a cumulative sum color image.
    
    Parameters:
    image (np.ndarray): The input image, which can be in grayscale or color.
    subOnDecrease (bool): Whether to subtract pixel value on decrease from previous pixel (for cumulative sum).
    
    Returns:
    tuple: 
        - Two numpy arrays containing the sum of pixel values along the y-axis (columns) and x-axis (rows).
        - The matrix with row numbers and final cumulative sums.
        - The output color image with cumulative sums distributed across channels.
    """
    logger.trace("Starting processImage function")
    logger.debug(f"Image shape: {image.shape}, subtract on decrease: {subOnDecrease}")

    # Convert the image to grayscale if it is not already
    height, width = image.shape[:2]
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.debug("Image converted to grayscale")
    else:
        gray = image
        logger.debug("Image is already in grayscale")
    
    # Sum the pixel values along the y-axis (sum along columns)
    col_sums = np.sum(gray, axis=0)
    logger.debug(f"Column sums: {col_sums}")

    # Sum the pixel values along the x-axis (sum along rows)
    row_sums = np.sum(gray, axis=1)
    logger.debug(f"Row sums: {row_sums}")

    logger.debug(f"Calculating Cumulative Sum of Image, subtract on decrease: {subOnDecrease}")
    
    # Compute the maximum sum a row could have
    max_sum = width * 255
    
    # Thresholds for each channel
    threshold1 = max_sum / 3
    threshold2 = 2 * threshold1
    
    # Initialize the cumulative sum and color image
    color_image = np.zeros((*gray.shape, 3), dtype=np.uint8)
    cumulative_sums = []

    pbar = tqdm(total=height * width, desc="Building Cumulative Sum of Image")
    for row in range(height):
        cumulative_sum = 0
        for col in range(width):
            pixel_value = gray[row, col]
            if col > 0:
                if gray[row, col] < gray[row, col - 1] and subOnDecrease:
                    cumulative_sum -= pixel_value
                else:
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
            pbar.update(1)
        cumulative_sums.append(cumulative_sum)
    pbar.close()
    
    cumulative_sums_matrix = np.array(cumulative_sums, dtype=np.int64)
    if DEBUG:
        cv2.imwrite(os.path.join(DEBUG_FOLDER, 'cumsum.jpg'), color_image)

    logger.debug(f"Finished processImage function")
    logger.debug(f"Returning col_sums: {col_sums.shape}, row_sums: {row_sums.shape}, cumulative_sums_matrix: {cumulative_sums_matrix.shape}, color_image: {color_image.shape}")
    return col_sums, row_sums, cumulative_sums_matrix, color_image

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

def process_images_in_folder(folder_path):

    # Supported image extensions
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    # Iterate through all files in the folder
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                image_path = os.path.join(root, file)
                print_image_info(image_path)

def removeCCOverThreshold(image, area_threshold):
    logger.debug(f"Removing Connected Components with area greater than: {area_threshold}.")
   # Check if the image is already grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    
    height, width = image.shape[:2]

    # Calculate white pixel percentage
    #wpa = np.count_nonzero(image_br == 255) / (height * width) * 100

    _, image_thresh = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY_INV)

    # Connected component analysis
    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(image_thresh)
    colored_labels_im = np.zeros_like(image)
    colored_labels_im = np.bitwise_not(colored_labels_im)

    # Iterate over labels
    areas = []
    for label in range(1, num_labels):  # Start from 1 to skip the background
        x, w, y, h, area = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_WIDTH], \
                            stats[label, cv2.CC_STAT_TOP], stats[label, cv2.CC_STAT_HEIGHT], \
                            stats[label, cv2.CC_STAT_AREA]

        # Color the pixels of the current label
        #logger.debug(f"Connected Component: {label}, x:{x}, w:{w}, y:{y}, h:{h}, area:{area}.")
        areas.append(area)
        if (area > area_threshold) or (h > height/4) or (w > width/4) :
            logger.debug(f"Connected Component: {label} exceeded threshold. x:{x}, w:{w}, y:{y}, h:{h}, area:{area}.")
            mask = (labels_im == label)
            colored_labels_im[mask] = 0

    logger.debug(f"image area: {height*width}, threshold({area_threshold}): {area_threshold} cc areas max: {max(areas)}, cc areas min: {min(areas)}.")
    #logger.debug(f"image ares: {np.sort(areas)[::-1]}.")
    
    return colored_labels_im

def group_and_average(values, threshold: int):
    """
    Groups values within a threshold and adds their average to a new list. 
    Single values not part of a group are added as is.

    Parameters:
    values (list of int or float): The list of values to process.
    threshold (int or float): The maximum distance between values to consider them as part of the same group.

    Returns:
    list: A new list containing the average of grouped values or individual values.
    """
    logger.debug(f"Processing values with threshold {threshold}")

    values = sorted(values)
    result = []
    group = [values[0]]

    for v in values[1:]:
        if v - group[-1] <= threshold:
            group.append(int(v))
        else:
            if len(group) > 1:
                avg = np.mean(group)
                result.append(int(avg))
            else:
                result.append(int(group[0]))
            group = [v]
    
    # Handle the last group
    if len(group) > 1:
        avg = np.mean(group)
        result.append(int(avg))
    else:
        result.append(int(group[0]))

    logger.debug(f"Resulting values: {result}")
    return result


import cv2
import numpy as np
import matplotlib.pyplot as plt
from segmentation import filterMinNConsecutive
from utils import util_img_cc_lbl_hm
from imageman import shadeByDistanceBetweenInk


input_image_path = r'C:\Users\User\Documents\E2CR\segmntation\seg_4159363_00361.jpg_006_009jpg'
in_image = cv2.imread(input_image_path)
x = util_img_cc_lbl_hm(in_image)
base_image = in_image.copy()

in_image = 255-in_image
height, width = in_image.shape[:2]


cv2.imshow("x",x)
cv2.waitKey()


def find_first_white_pixel_y(image):
    # Get the dimensions of the image
    height, width = image.shape[:2]
    print(image.shape)
    print(f"width: {width}, height:{height}")



def plot_histograms(sum_y, sum_x):
    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot the horizontal histogram (sum along y-axis)
    std = np.std(y_values)
    axs[0].barh(range(len(sum_y)), sum_y, color='gray', edgecolor='black')
    axs[0].set_xlabel('Sum of pixel values')
    axs[0].set_ylabel('Y-axis')
    axs[0].set_ylim(0,len(sum_y))
    axs[0].invert_yaxis()
    axs[0].axhline(y=average_y_value, color='green', linestyle='--', linewidth=1)
    axs[0].axhline(y=np.argmax(sum_y)-std, color='yellow', linestyle='--', linewidth=2)
    axs[0].axhline(y=np.argmax(sum_y), color='blue', linestyle='--', linewidth=2)
    axs[0].axhline(y=np.argmax(sum_y)+std, color='yellow', linestyle='--', linewidth=2)
    axs[0].legend()
    axs[0].set_title('Histogram of Pixel Value Sums Along Rows')

    # Plot the vertical histogram (sum along x-axis)
    axs[1].bar(range(len(sum_x)), sum_x, color='gray', edgecolor='black')
    axs[1].axhline(y=np.mean(sum_x), color='blue', linestyle='--', linewidth=2)
    axs[1].axhline(y=np.mean(sum_x)-np.std(sum_x), color='green', linestyle='--', linewidth=2)
    axs[1].set_xlabel('X-axis')
    axs[1].set_ylabel('Sum of pixel values')
    axs[1].set_title('Histogram of Pixel Value Sums Down Columns')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()



# Plot the histograms
plot_histograms(rowsums, colsums)

max = np.argmax(rowsums)
#max = np.mean(y_values)

std = np.std(y_values)
top = max-std
print(max,std)

in_image = cv2.line(in_image, (0, int(max-std)-1), (width, int(max-std)-1), (0,0,255), thickness=1)
in_image = cv2.line(in_image, (0, int(max)), (width, int(max)), (255,0,0), thickness=2)
in_image = cv2.line(in_image, (0, int(max+std)+1), (width, int(max+std)+1), (0,0,255), thickness=1)

cv2.imshow("sliced points", in_image)
cv2.waitKey()

print(int(max-(std)),int(max+(std)))
sliced_image = base_image[int(max-(std)):int(max+(std)), :]
sliced_image = np.bitwise_not(sliced_image)
sliced_image[sliced_image <= 245] = 0

cv2.imshow("sliced_image", sliced_image)
cv2.waitKey()

colsums, rowsums = sumColRows(sliced_image)

mean = np.mean(colsums)
std = np.std(colsums)

# Plot the histograms
plot_histograms(rowsums, colsums)

y_lowpoints = []
for i, value in enumerate(colsums):
    if value == 0:
        y_lowpoints.append(i)
        
y_lowpoints = filterMinNConsecutive(y_lowpoints,5)


for point in y_lowpoints:
    in_image = cv2.line(in_image, (point,0) , (point, height), (0,0,255), thickness=2)
