import cv2
#import os
#import matplotlib.pyplot as plt
from common import *
from imageman import preProcessImage, crop_between_lines2, draw_path_on_image, cropTextRows
from segmentation import findPaths, colTransitionPoints
from loguru import logger


def save_crops(images):
    """
    Display the list of images in a grid.

    Args:
        images (list of np.array): List of images to display.
    """
    num_images = len(images)
    cols = 3
    rows = (num_images + cols - 1) // cols

    for i, image in enumerate(images):
        crop_path = os.path.join(os.getcwd(), "output", f"_crop_{i}.png")
        cv2.imwrite(crop_path, image)

# Paths
image_path = 'E:/E2CR/sample_images_for_ocr/R. 317 (7).jpg'

logger.info(f"Loading image from {image_path}")

imageToSegment = cv2.imread(image_path)

# Preprocessing
ppImage = preProcessImage(imageToSegment.copy())
cv2.imwrite(os.path.join(os.getcwd(), 'debug','segmentation', 'ppImage.jpg'), ppImage)

paths_found = findPaths(ppImage,threshRate=.85,eps=10)
pathsOnImage = imageToSegment.copy()
for path in paths_found:
    pathsOnImage = draw_path_on_image(pathsOnImage,path)
cv2.imwrite(os.path.join(os.getcwd(), 'debug','segmentation', 'pathsOnImage.jpg'), pathsOnImage)

if imageToSegment is not None:
    cropped_images = crop_between_lines2(ppImage, paths_found)
    
    for num, text_line in enumerate(cropped_images):
        
        word_breaks = colTransitionPoints(text_line)
        '''
        h, w = text_line.shape
        text_line = cv2.cvtColor(text_line, cv2.COLOR_GRAY2BGR)
        for x in transfound_found:

            text_line = cv2.line(text_line, (x, 0), (x, h), Colors.BLUE, thickness=2)
            cv2.putText(text_line, f"Xt: {x}", (x+2, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, Colors.BLUE, 2)
        
        '''
        words = cropTextRows(text_line,word_breaks)
        for w, word in enumerate(words):
            print("Should be saving here")
            print(cv2.imwrite(os.path.join(os.getcwd(), 'output', f'text_line_{num}_word{w}.jpg'), word ))

    #save_crops(cropped_images)
else:
    logger.error("Failed to load the image")



