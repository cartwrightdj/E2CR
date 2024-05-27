import cv2
import numpy as np
from heapq import heappop, heappush
import matplotlib.pyplot as plt
import os
from imageman_manipulation import *
from segmentation import *



# Create the output directory if it doesn't exist
cropped_output_dir = "E:/E2CR_NM/output/"
os.makedirs(cropped_output_dir, exist_ok=True)

# Paths
image_path = 'E:/E2CR_NM/sample_images_for_ocr/onetime/4159363_00373.jpg'
thresholded_image_path = "E:/E2CR_NM/output/thresholded_image.jpg"
output_path = "E:/E2CR_NM/output/shortest_path_output_final.jpg"

# Processing

baseImage = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
original_image = baseImage.copy()

#PreProccessing
ppImage = preProcessImage(baseImage)
cv2.imwrite(f"E:/E2CR_NM/output/ppImage.jpg", ppImage)


imageHist = getImageHist(ppImage, ROW_HIST,125)
breakPointThresh = max(imageHist) * .90 #threshRate
peaks = find_high_points(imageHist, breakPointThresh)
#for peak in peaks:
#    ppImage = draw_line(ppImage, peak, orientation='horizontal', color=(0, 0, 255), thickness=2)

clusters, labels = group_by_proximity(peaks, 20, 1)
peaks, _ = find_breakpoints_max(clusters, imageHist)

for peak in peaks:
    ppImage = draw_line(ppImage, peak, orientation='horizontal', color=(0, 255, 0), thickness=1)


gradients = calculate_gradients(ppImage)


#output_image = original_image
output_image = ppImage

thresh_image = cv2.bitwise_not(ppImage)

paths_found=[]
#peaks = [100,200,300,400,500,1000,1500]
for peak in peaks:
    shortest_path, max_y, min_y = find_shortest_path(thresh_image, gradients, peak, log_cost_factor=100, bias_factor=10, gradient_factor=5)  # Adjust factors as needed
    paths_found.append(shortest_path)
    output_image = draw_path_on_image(output_image, shortest_path, max_y, min_y)
    #print(cv2.imwrite(f"E:/E2CR_NM/output/result_{peak}.jpg", output_image))
 
print(cv2.imwrite(f"E:/E2CR_NM/output/result_final.jpg", output_image))

height, width = output_image.shape[:2]



def crop_between_lines(image, lines):
    """
    Crop areas between each pair of generally horizontal lines,
    from the top to the first line, and from the last line to the bottom of the page.
    Return a list of cropped images with only the area between the lines included.

    Args:
        image (np.array): The input image.
        lines (list of list of tuples): List of lines, where each line is a list of (y, x) points.

    Returns:
        list of np.array: List of cropped images between the lines.
    """
    logger.info("Starting crop between lines")

    # Ensure the image is not empty
    if image is None or image.size == 0:
        logger.critical("Input image is empty or invalid")
        raise ValueError("Input image is empty or invalid")

    cropped_images = []

    # Convert to grayscale if the input is a color image
    if len(image.shape) == 3:
        logger.debug("Converting color image to grayscale")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    height, width = gray_image.shape
    logger.debug(f"Image dimensions: height={height}, width={width}")

    masked_image = image


    for i in range(len(lines)):
            mask_pts = []
            if i == 0:
                logger.debug(f"Processing from top of the image to line {i}")
                y_start = 0
                _, y_end = getLineStats(lines[i])
                mask_pts.extend([[0, 0], [width, 0]])
                mask_pts.extend([[x, y] for y, x in lines[i]])
            elif i == len(lines) - 1:
                logger.debug(f"Processing between line {i-1} and bottom of the image")
                _, y_start = getLineStats(lines[i])
                y_end = height
                mask_pts.extend([[x, y] for y, x in lines[i]])
                mask_pts.extend([[width, height], [0, height]])
            else:
                logger.debug(f"Processing between line {i-1} and line {i}")
                y_start, _ = getLineStats(lines[i-1])
                _, y_end = getLineStats(lines[i])
                mask_pts.extend([[x, y] for y, x in lines[i-1]])
                mask_pts.append([width, y_start])
                mask_pts.append([width, y_end])
                mask_pts.extend([[x, y] for y, x in lines[i]])
                mask_pts.append([0, y_end])
                mask_pts.append([0, y_start])

            logger.trace(f"Mask points for segment {i}: {mask_pts}")

            # Create a mask for the area between the current line and the next line
            mask = np.zeros_like(gray_image, dtype=np.uint8)
            pts = np.array(mask_pts, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

            # Apply the mask to the original image to extract the area between the lines
            masked_image = cv2.bitwise_and(gray_image, gray_image, mask=mask)

            # Crop the bounding box of the masked region
            cropped_image = masked_image[y_start:y_end, 0:width]

            logger.debug(f"Crop from y_start={y_start}, to y_end={y_end}")

            # Ensure the cropped image is not empty
            if cropped_image.size > 0:
                cropped_images.append(cropped_image)
            else:
                logger.warning(f"Empty crop region: y_start={y_start}, y_end={y_end}")

    logger.info("Finished cropping between lines")
    return cropped_images

def display_images(images):
    """
    Display the list of images in a grid.

    Args:
        images (list of np.array): List of images to display.
    """
    import matplotlib.pyplot as plt
    num_images = len(images)
    cols = 3
    rows = (num_images + cols - 1) // cols

    plt.figure(figsize=(15, 5 * rows))

    '''for i, img in enumerate(images):
        if img.size > 0:  # Ensure the image is not empty
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(f'Cropped Image {i + 1}')
            plt.axis('off')

    plt.tight_layout()
    plt.show()'''

    for i, image in enumerate(images):
        denoise_image_path = os.path.join(os.getcwd(),"output", f"_crop_{i}.png")
        cv2.imwrite(denoise_image_path, image)



logger.info(f"Loading image from {image_path}")
original_image = cv2.imread(image_path)

if original_image is not None:
    cropped_images = crop_between_lines(original_image, paths_found)
    display_images(cropped_images)
else:
    logger.error("Failed to load the image")


