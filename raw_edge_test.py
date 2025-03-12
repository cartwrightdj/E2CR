

image_path = r'C:\Users\User\Documents\PythonProjects\E2CR\debug\4159363_00361.jpg_pp_denoised.tiff'

import cv2
import numpy as np
import matplotlib.pyplot as plt

def edge_detection(image_path, method="sobel"):
    """
    Detects edges in an image using Sobel, Canny, or Laplacian method.
    
    Parameters:
    - image_path (str): Path to the input image.
    - method (str): The edge detection method. Options: "sobel", "canny", "laplacian".
    
    Returns:
    - edge_image (numpy.ndarray): The detected edge map.
    """
    
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    
    if method == "sobel":
        # Apply Sobel edge detection (gradient-based)
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in X direction
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in Y direction
        edge_image = np.sqrt(sobelx**2 + sobely**2)  # Compute gradient magnitude
        edge_image = (edge_image / edge_image.max() * 255).astype(np.uint8)  # Normalize to 0-255

    elif method == "canny":
        # Apply Canny edge detection (automatic threshold-based)
        edge_image = cv2.Canny(image, 50, 150)
         # Find contours
        contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a blank black image (same size as original)
        contour_image = np.zeros_like(image)

        # Draw contours on the blank image
        cv2.drawContours(contour_image, contours, -1, (255, 255, 255), thickness=2)
        cv2.imshow('contours',contour_image)

        # Apply the mask on the original image
        result_image = cv2.bitwise_and(image, image, mask=contour_image)
        #result_image[result_image != 0] = 255
        cv2.imshow('result_image',result_image)

    elif method == "laplacian":
        # Apply Laplacian edge detection (second derivative-based)
        edge_image = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
        edge_image = np.abs(edge_image)  # Take absolute values to remove negative edges
        edge_image = (edge_image / edge_image.max() * 255).astype(np.uint8)  # Normalize

    else:
        raise ValueError("Invalid method! Choose from 'sobel', 'canny', or 'laplacian'.")

    return edge_image

# Example Usage

edge_methods = ["sobel","canny","laplacian"]

# Plot all edge detection methods
plt.figure(figsize=(12, 4))
for i, method in enumerate(edge_methods):
    edges = edge_detection(image_path, method)
    
    plt.subplot(1, 3, i + 1)
    plt.imshow(edges, cmap="gray")
    plt.title(f"{method.capitalize()} Edges")
    plt.axis("off")

plt.tight_layout()
plt.show()

