import cv2
from common import DEBUG_FOLDER
from imageman import preProcessImage, draw_path_on_image
from segmentation import findTextSeperation
import os
import numpy as np

image_path = r'E:\E2CR\debug\segmentation\cs_image.jpg'
imageToTest = cv2.imread(image_path)

# Summing the values along the third dimension (axis=2)
summed_array = np.sum(imageToTest, axis=2)
# Normalize the summed array
min_val = np.min(summed_array)
max_val = np.max(summed_array)
normalized_array = (summed_array - min_val) / (max_val - min_val)

# Scale to range [0, 255] and convert to uint8
image_array = (normalized_array * 255).astype(np.uint8)


image_path = 'E:/E2CR/sample_images_for_ocr/4159363_00363.jpg'
imageToSegment = cv2.imread(image_path)

ppImage = preProcessImage(imageToSegment.copy())

# Summing the values along the third dimension (axis=2)
summed_array = np.sum(imageToTest, axis=2)
# Normalize the summed array
#min_val = np.min(summed_array)
#max_val = np.max(summed_array)
#normalized_array = (summed_array - min_val) / (max_val - min_val)
image_array = image_array[:, ::-1]
combined = cv2.bitwise_not(image_array * cv2.bitwise_not(ppImage))
cv2.imwrite(os.path.join(DEBUG_FOLDER, 'combined_image.jpg'), combined)

# Calculate the sum of each row
row_sums = np.sum(ppImage, axis=1)

# Set all elements in each row to the sum of that row
new_array = np.zeros_like(ppImage)
for i in range(ppImage.shape[0]):
    new_array[i, :] = row_sums[i]
cv2.imwrite(os.path.join(DEBUG_FOLDER, 'summed_array.jpg'), new_array)

combined = cv2.bitwise_not(new_array * cv2.bitwise_not(ppImage))


paths_found = findTextSeperation(combined, method='y_average',threshRate=94)
    
pathsOnImage = imageToTest.copy()
for path in paths_found:
    pathsOnImage = draw_path_on_image(new_array,path)
    cv2.imwrite(os.path.join(DEBUG_FOLDER, 'pathsOnImage_new_array.jpg'), pathsOnImage)
for path in paths_found:
    pathsOnImage = draw_path_on_image(imageToSegment,path)
    cv2.imwrite(os.path.join(DEBUG_FOLDER, 'baseImage_cumsum.jpg'), pathsOnImage)