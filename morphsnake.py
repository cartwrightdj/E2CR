import cv2
import numpy as np
import morphsnakes
import matplotlib.pyplot as plt

# Load image
image = cv2.imread(r'C:\Users\User\Documents\PythonProjects\E2CR\segmentation\row_4159363_00361.jpg_009.tiff', cv2.IMREAD_GRAYSCALE)

image = cv2.bitwise_not(image)



# Step 1: Preprocessing - Apply Gaussian Blur
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Step 2: Compute Edge Indicator Function (Sobel for better edge response)
dx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
dy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = np.sqrt(dx**2 + dy**2)
gradient_magnitude = gradient_magnitude / gradient_magnitude.max()  # Normalize

# Reduce the effect of horizontal shrinking by scaling dx
dx = dx * 0.5  # Reduce sensitivity to horizontal edges (less shrinking horizontally

# Convert gradient into edge-stopping function
gI = 1.0 / (1.0 + gradient_magnitude)  

# Step 3: Merge Close Contours Using Morphological Operations
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# **Dilate** to merge nearby contours (adjust kernel size for larger gaps)
kernel = np.ones((5, 5), np.uint8)  # Larger kernel → more merging
binary = cv2.dilate(binary, kernel, iterations=2)  

# Step 4: Initialize the Contour **Around Larger Text Regions**
initial_ls = np.zeros_like(binary)
initial_ls[binary > 0] = 1  # Start from high-intensity regions

# **Invert the initial level set to start the snake from outside**
initial_ls = np.ones_like(binary)  # Start with everything active
initial_ls[binary > 0] = 0  # Remove only text areas
initial_ls[50:-50, 50:-50] = 1  # Keep a border active to start "in between"

cv2.imshow('binary',binary)
cv2.waitKey()

# Manual iteration counter
iteration_count = [0]

# Function to visualize iterations
def visualize_callback(phi):
    iteration = iteration_count[0]
    plt.clf()
    plt.imshow(image, cmap="gray")
    plt.contour(phi, levels=[0], colors="r")  # Active contour in red
    plt.title(f"Iteration {iteration}")
    plt.pause(0.5)
    iteration_count[0] += 1  # Increment iteration count

# Step 5: Apply Morphological Snakes with Merging Tuning
msnake = morphsnakes.morphological_geodesic_active_contour(
    gI, iterations=10, init_level_set=initial_ls, smoothing=1, balloon=-1, iter_callback=visualize_callback
)

# Convert final segmentation to uint8
msnake = np.array(msnake, dtype=np.float64)
msnake = (msnake - msnake.min()) / (msnake.max() - msnake.min())  # Normalize to [0,1]
segmented_text = (msnake * 255).astype(np.uint8)

# Show final result
plt.figure()
plt.imshow(segmented_text, cmap="gray")
plt.title("Final Segmentation - Merged Contours")
plt.show()
