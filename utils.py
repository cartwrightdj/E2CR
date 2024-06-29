'''Example:
Script Name: utils.py /.utils.py

Author: David J. Cartwright]
Date: 5/25/2024
Last Modification Date:

Dependencies:
    - os (standard library)
'''
import cv2
import os
from common import logger, DEBUG, DEBUG_FOLDER
import numpy as np
from tqdm import tqdm
import csv
from scipy.stats import entropy

def unique_filepath(path):
    """
    Ensure necessary folders are created for the given path. If it's a file path,
    handle the situation where the file already exists by appending a number at the end.

    Args:
        path (str): The desired file or folder path.

    Returns:
        str: The unique file path with necessary folders created.
    """
    # Check if the path is a file or a directory
    base, extension = os.path.splitext(path)
    if extension:  # It's a file path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # Check if file already exists and create a unique file path
        counter = 1
        new_filepath = path
        while os.path.exists(new_filepath):
            new_filepath = f"{base}_{counter}{extension}"
            counter += 1
        
        return new_filepath
    else:  # It's a directory path
        os.makedirs(path, exist_ok=True)
        return path
    
def util_img_cc_lbl_hm(image, minarea=0,operation="util_img_cc_lbl_hm"):
    # Check if the image is already grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        logger.warning("This function received a non-grayscale image, results will not be accurate")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    height, width = image.shape[:2]

    # Calculate white pixel percentage
    #wpa = np.count_nonzero(image == 255) / (height * width) * 100

    _, image_thresh = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY_INV)

    # Connected component analysis
    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(image_thresh)
    logger.debug(f"Found {num_labels} Connected Components")

    # Create colored labels image with 3 channels for RGB
    colored_labels_im = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Generate random colors for each label
    colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)

    # Iterate over labels
    pbar = tqdm(total=num_labels-1, desc=f"Creating heat map from {num_labels} labels.")
    label_stats = []
    for label in range(1, num_labels):  # Start from 1 to skip the background
        x, y, w, h, area = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], \
                           stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT], \
                           stats[label, cv2.CC_STAT_AREA]
        
        component_mask = (labels_im == label).astype(np.uint8)
        # Extract the pixel values of the component
        component_pixels = image[component_mask == 1]
        # Calculate the entropy of the component pixels
        ent = ccEntropy(component_pixels)
        proprat = max(h/w,w/h)
        filled_area = "{:.2f}".format(area/(h*w))
        size = w * h
        label_stats.append([label, x, w, y, h,proprat, area,size,filled_area,  ent])

        if area > minarea:
            # Color the pixels of the current label
            
            mask = (labels_im == label)
            colored_labels_im[mask] = colors[label]

            # Add label number to the image
            B, G, R = int(colors[label][0]), int(colors[label][1]), int(colors[label][2])
            if h > 20 or w > 20:
                colored_labels_im = cv2.putText(colored_labels_im, f"{label} {area} (x:{x},y:{y})", 
                                            (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (B, G, R), 1)
            
        pbar.update(1)

    pbar.desc = pbar.desc + " Complete."
    pbar.close()

    if DEBUG: 
        util_csv_debug(label_stats, 'util_img_cc_lbl_hm', labels=['label', 'x', 'w', 'y', 'h','proprat', 'area','size','filled_area',  'entropy'])
        cv2.imwrite(os.path.join("./output",f"{operation}.tiff"), colored_labels_im)

    return colored_labels_im

def util_img_diff(image1, image2):
    """
    Compute the absolute difference between two images.
    
    Parameters:
        image1 (numpy.ndarray): The first input image.
        image2 (numpy.ndarray): The second input image.
    
    Returns:
        numpy.ndarray: An image showing the absolute difference between image1 and image2.
    """
    # Ensure both images have the same size
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Compute absolute difference
    diff_image = cv2.absdiff(image1, image2)

    return diff_image

def util_contour_hm(image):
    # Load the image

    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = image.shape[:2]

    # Apply threshold to the image
    thresh = 100
    ret, thresh_img = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    logger.info(f"Found: {len(contours)} contours.")

    # Draw the contours on the original image
    # = np.zeros_like(image)
    #img_contours = np.bitwise_not(img_contours)
    img_contours = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    colors = np.random.randint(0, 255, size=(len(contours), 3), dtype=np.uint8)
    
    buffer = 25
    label_stats = []
    img_contours = cv2.cvtColor(img_contours, cv2.COLOR_GRAY2BGR)
    for i, cnt in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        hwratio = round(w/h,2) if w > h else round(h/w,2)
        #isclosed = cv2.isContourConvex(cnt) and np.array_equal(cnt[0], cnt[-1])
        isclosed = np.array_equal(cnt[0], cnt[-1])
        perimeter = cv2.arcLength(cnt,isclosed)
        if ((width-buffer < x) or (x < buffer)) and ((height-buffer < y) or (y < buffer)) or (width-buffer < x+w) or (height-buffer < y+h) or ((hwratio > 6) and area > 3000):


            logger.info(f"Contour: {i} x:{x}, w:{w}, y:{y}, h:{h}, area:{area}, hw ratio:{hwratio}, is closed:{isclosed}, perimeter: {perimeter}")
            B, G, R = int(colors[i][0]), int(colors[i][1]), int(colors[i][2])
            cv2.drawContours(img_contours, [cnt], -1, (B, G, R), 1)
            label_stats.append([i,x,w,y,h,area,hwratio,isclosed,perimeter])
            img_contours = cv2.putText(img_contours, f"{i} ({x,y}) ({area})", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (B, G, R), 1)

    logger.info(f"i: {i} contours.")
    if DEBUG: util_csv_debug(label_stats,'util_contour_hm',labels=['label','x','w','y','h','area','hw ratio','is closed','perimeter'])

    return img_contours

def util_csv_debug(data,operation_name='DEBUG',labels=None):
    logger.info(f"Writing: {len(data)} to csv file.")
    with open(os.path.join('./debug',f'{operation_name}.csv'),'a', newline='') as csvfile:
        write = csv.writer(csvfile)
        if labels is not None:
            write.writerow(labels)    
        write.writerows(data)


def ccEntropy(component_pixels):
    """Calculate the entropy of the pixel values in a connected component."""
    # Compute histogram of pixel values
    hist, _ = np.histogram(component_pixels, bins=256, range=(0, 255))
    # Normalize the histogram
    hist = hist / np.sum(hist)
    # Calculate entropy
    ent = entropy(hist, base=2)
    return ent



