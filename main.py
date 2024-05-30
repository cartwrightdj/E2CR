import cv2
import os
#import matplotlib.pyplot as plt
from common import DefaultParameters, Statistics, DEBUG_FOLDER
from imageman import preProcessImage, cropRowsFromImage, draw_path_on_image, cropTextFromRow, percentWhite, crop_image_to_content, process_images_in_folder
from segmentation import findTextSeperation, colTransitionPoints
from loguru import logger


# Paths
image_path = 'E:/E2CR/sample_images_for_ocr/R. 317 (2).jpg'
imageToSegment = cv2.imread(image_path)

def getTextFromImage(image_path: str):
    Parameters = DefaultParameters()
    logger.info(f"Loading image from {image_path}")
    imageToSegment = cv2.imread(image_path)

    # Preprocessing
    ppImage = preProcessImage(imageToSegment.copy(),Parameters)
    cv2.imwrite(os.path.join(DEBUG_FOLDER, '0_PreProcessedImage.jpg'), ppImage)
    
    paths_found = findTextSeperation(ppImage)
    
    pathsOnImage = imageToSegment.copy()
    for path in paths_found:
        pathsOnImage = draw_path_on_image(pathsOnImage,path)
    cv2.imwrite(os.path.join(DEBUG_FOLDER, 'pathsOnImage.jpg'), pathsOnImage)

    if imageToSegment is not None:
        RowsOfText = cropRowsFromImage(ppImage, paths_found)
        
        for num, RowOfText in enumerate(RowsOfText):
            
            word_breaks = colTransitionPoints(RowOfText)
            line_split_file = RowOfText.copy()
            for x in word_breaks:
                height, width = line_split_file.shape
                # Draw a vertical line from (x_value, 0) to (x_value, height)
                line_split_file = cv2.line(line_split_file, (x, 0), (x, height), (0, 255, 0), 2)
                cv2.imwrite(os.path.join(os.getcwd(), 'output', f'line_{num:03d}_splits.jpg'), line_split_file )

            words = cropTextFromRow(RowOfText,word_breaks)
            for w, word in enumerate(words):
                crop = crop_image_to_content(word)
                height, width = crop.shape
                if not (((percentWhite(word) > 90) or (width < 26) or (height < 10))): 
                    cv2.imwrite(os.path.join(os.getcwd(), 'output', f'line_{num:03d}_word_{w:03d}.jpg'), crop )

    else:
        logger.error("Failed to load the image")


getTextFromImage(image_path)

DefaultParameters.print_defaults    
Statistics.display()

