import cv2
from segment import Segment
import numpy as np
import matplotlib.pyplot as plt

image_path = r'C:\Users\User\Documents\E2CR\segmentation\4159363_00361_Result.tiff'

img = cv2.imread(image_path)

Doc = Segment(image=img)
print(Doc)
#cc = Doc.extract_cc()
Doc._filterIndexByValThreshold(.95)

Doc._calculateTextRowHeight()
Doc.fibvtShow()



