import cv2
#import os
#import matplotlib.pyplot as plt
from common import *
from imageman import preProcessImage, cropRowsFromImage, draw_path_on_image, cropTextFromRow, percentWhite, crop_image_to_content
from segmentation import findTextSeperation, colTransitionPoints
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
image_path = 'E:/E2CR/sample_images_for_ocr/R. 317 (9).jpg'
Statistics.test = 2

logger.info(f"Loading image from {image_path}")

def getTextFromImage(image_path: str):
    imageToSegment = cv2.imread(image_path)

    # Preprocessing
    ppImage = preProcessImage(imageToSegment.copy(),pp_config)
    cv2.imwrite(os.path.join(os.getcwd(), 'debug','segmentation', 'ppImage.jpg'), ppImage)

    paths_found = findTextSeperation(ppImage,seg_config)
    pathsOnImage = imageToSegment.copy()
    for path in paths_found:
        pathsOnImage = draw_path_on_image(pathsOnImage,path)
    cv2.imwrite(os.path.join(os.getcwd(), 'debug','segmentation', 'pathsOnImage.jpg'), pathsOnImage)

    if imageToSegment is not None:
        RowsOfText = cropRowsFromImage(ppImage, paths_found)
        
        for num, RowOfText in enumerate(RowsOfText):
            
            word_breaks = colTransitionPoints(RowOfText)
            for x in word_breaks:
                height, width = RowOfText.shape
                # Draw a vertical line from (x_value, 0) to (x_value, height)
                RowOfText = cv2.line(RowOfText, (x, 0), (x, height), (0, 255, 0), 2)
                cv2.imwrite(os.path.join(os.getcwd(), 'output', f'line_{num:03d}_splits.jpg'), RowOfText )

            words = cropTextFromRow(RowOfText,word_breaks)
            for w, word in enumerate(words):
                crop = crop_image_to_content(word)
                height, width = crop.shape
                if not (((percentWhite(word) > 90) or (width < 26) or (height < 10))): 
                    cv2.imwrite(os.path.join(os.getcwd(), 'output', f'line_{num:03d}_word_{w:03d}.jpg'), crop )

    else:
        logger.error("Failed to load the image")


getTextFromImage(image_path)

RuntimeParameters.display()      
Statistics.display()

