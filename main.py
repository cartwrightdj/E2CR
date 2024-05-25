
import os
directory = 'E:/E2CR/xyz'

# Import required packages
from e2cr_plotting import *
from segmentation import *
from e2cr_plotting import *
from imageman_manipulation import *
from common import *
import cv2
from PIL import Image
import tempfile
import numpy as np
import os
from loguru import logger
import sys

logger.remove()  # Remove the default logger
logger.add(sys.stderr, level="TRACE")

def TryRecursion(baseImage,imagename = "ScannedImage"):
    row_num = -1

    def find_text_rows(Image, level=0, threshRate = .96):
        
        nonlocal row_num
        rows = []
        
        #threshRate = .96  # (higher generally less clips)
        Image = remove_small_black_areas(Image, 95)
        imageHist = getImageHist(Image, ROW_HIST)
        breakPointThresh = max(imageHist) * threshRate
        peaks = find_high_points(imageHist, breakPointThresh)
        clusters, labels = group_by_proximity_dbscan(peaks, 40, 1)
        peaks = find_breakpoints_average(clusters, imageHist)
        image_with_lines = draw_horizontal_lines(Image, peaks)
        crops, num_skipped = crop_image(Image, peaks, CROP_HORIZONTAL)

        if len(crops) != 1:
            for crop in crops:
                sub_crop = find_text_rows(crop, level + 1,threshRate-.02)                
                rows.extend(sub_crop)  # Collect rows from sub-crops              
        else:
            if threshRate < .50: 
                logger.warning(f"Minimum recursion threshRate: {round(threshRate,2)} reached")
                if logger.level == "TRACE":
                    cv2.imwrite(f"e:/E2CR/crops/_crops[0].jpg", crops[0])
                rows.append(find_text_words(crops[0])) 
                row_num += 1
                return rows

            h, w = crops[0].shape[:2]
            if h > DefaultSeg.EXPECTED_IMAGE_HW * 2:
                logger.trace(f"While finding text rows, a row that was indicated to be complete but exceeded twice the expected height")
                
                sub_crop = find_text_rows(crops[0], level + 1,threshRate-.02) 
                rows.extend(sub_crop)  # Collect rows from sub-crops
            else: 
                rows.append(find_text_words(Image))  # Add the final crop to rows
                row_num += 1

        return rows
    
    def find_text_words(text_row,threshRate = .95): 
        nonlocal row_num
        nonlocal imagename

        words  = []
        cv2.imwrite(f"e:/E2CR/crops/{imagename}_textrow_{row_num+1}.jpg", text_row)
        imageHist = getImageHist(text_row,COL_HIST)
        breakPointThresh = max(imageHist) * threshRate
        peaks = find_high_points(imageHist,breakPointThresh)
        clusters, labels = group_by_proximity_dbscan(peaks,40,1)
        #peaks = find_breakpoints_average(clusters,imageHist)
        peaks, _ = find_breakpoints_max(clusters,imageHist)
        image_with_lines = draw_vertical_lines(text_row,peaks)
        print(cv2.imwrite(create_unique_filepath(os.path.join(cwd,'output',f'{imagename[0:-4]}','text_rows',f'{imagename}_{row_num}_breaks.jpg')), image_with_lines))
        crops, num_skipped = crop_image(text_row,peaks,CROP_VERTICLE)
        for w, text_word in enumerate(crops):
            words.append(text_word)
                   
        return words
    
    return find_text_rows(baseImage)

print(f"cwd = {os.getcwd()}")

directory = os.path.join(cwd,'sample_images_for_ocr','onetime')

print(f"directory = {directory}")

for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".py"): 
        print(f"processing file: {filename}")
        baseImage = cv2.imread(os.path.join(directory, filename))
        ppImage = preProcessImage(baseImage)
        results = TryRecursion(ppImage,os.path.basename(filename))
        for i, row in enumerate(results):
            for y, text in enumerate(row):
                text = crop_to_non_white_areas(text)
                text = resize_image(text, target_size=(128, 32))
                cv2.imwrite(create_unique_filepath(os.path.join(cwd,'output',f'{filename[:-4]}','text_words',f'{filename[0:-4]}_{i:03d}_{y:03d}.jpg')), text)
                text = set_image_dpi(os.path.join(directory, filename))
        final_image, box_coords = combine_images(results)        
        cv2.imwrite(create_unique_filepath(os.path.join(cwd,'output',f"{filename[:-4]}_ppmarked_.jpg")), final_image)
        cv2.imwrite(create_unique_filepath(os.path.join(cwd,'output',f"{filename[:-4]}_marked_.jpg")), draw_boxes(baseImage, box_coords))
    
            
      