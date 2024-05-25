# Import required packages
from segmentation import *
import cv2
import pytesseract

import tempfile
import numpy as np
import os


print(cwd)


useDenoise = True




applyErode = False

#erodeKernel = np.ones((6, 6), np.uint8) 


USE_ADAPTIVE_THRESH = True
minHistSum = 1000   #if a rectangle contains less than this sum of pixel values, the image part will not be cropped out.
 
# Resize to 300dpi





            



def ocr_main():
    try:
        #baseImage = cv2.imread(r"E:\OCR\base_images\baseImage_1.png")
        baseImage = cv2.imread(r"E:\OCR\gray_denoise.png")
        chImage = baseImage.copy()
    except cv2.error as e:
        print(f"Error{e}")

    #ppImage = preProcessImage(baseImage, True, USE_ADAPTIVE_THRESH)
    findHistSplit(baseImage)
    #rectsOnImage, convexHullsOnImage = ocrOnFile(ppImage, baseImage)

    #cv2.imwrite(f"E:/OCR/bounding_rects.jpg", rectsOnImage)
    #cv2.imwrite(f"E:/OCR/bounding_convex_hulls.jpg", convexHullsOnImage)


if __name__ == "__main__":
    ocr_main()