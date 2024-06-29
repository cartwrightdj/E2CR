import cv2
import numpy as np
import matplotlib.pyplot as plt
from imageutils import shadeByDistanceBetweenInk

# Define the image path
image_path = r'C:\Users\User\Documents\E2CR\segmentation\seg_4159363_00361.jpg_006_009.tiff'

# Load the image in grayscale
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply shading function
shadedImage = shadeByDistanceBetweenInk(image, mode='cols')

# Calculate the sum of pixel values along each column
column_sums = np.sum(image, axis=0)

# Get the width of the image
width = image.shape[1]

# Calculate the mean and standard deviation of the column sums
mean_x = np.mean(column_sums)
std_x = np.std(column_sums)

# Create subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), constrained_layout=True)

# Plot the original image
ax1.imshow(image, cmap='gray', aspect='auto')
ax1.set_title('Image')
ax1.axis('off')

# Plot the column sums
ax2.plot(column_sums, color='black')
ax2.set_title('Column Sums')
ax2.set_xlabel('Column Index')
ax2.set_ylabel('Sum of Pixel Values')
ax2.set_xlim([0, width])

# Plot the mean line
ax2.axhline(y=mean_x, color='blue', linestyle='--', label='Mean x')
# Plot the standard deviation lines
ax2.axhline(y=mean_x + std_x, color='green', linestyle='--', label='Mean x + 1 STD')
ax2.axhline(y=mean_x - std_x, color='green', linestyle='--', label='Mean x - 1 STD')
ax2.legend()

# Plot the shaded image
ax3.imshow(shadedImage, cmap='gray', aspect='auto')
ax3.set_title('Shaded Image')
ax3.axis('off')

# Ensure the x-axis alignment is correct by setting the same limits
plt.show()
