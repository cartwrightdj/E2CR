import cv2
import numpy as np
import matplotlib.pyplot as plt
from e2cr import extract_txt_by_contour_mask, visualize_cc, extract_txt_by_contour_mask_sobel, crop_to_border, denoise, sharpen

def apply_contrast_filters(image_path):
    """
    Applies different contrast enhancement filters to an image and displays the results.
    
    Parameters:
    - image_path (str): Path to the input image.
    
    Returns:
    - Dictionary containing contrast-enhanced images.
    """
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 1️⃣ Histogram Equalization (Global)
    hist_equalized = cv2.equalizeHist(image)

    # 2️⃣ CLAHE (Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    adaptive_hist_equalized = clahe.apply(image)

    # 3️⃣ Gamma Correction (Adjust for Brightness)
    gamma = 1.5  # Increase for brighter image, decrease (<1) for darker
    gamma_corrected = np.power(image / 255.0, gamma) * 255.0
    gamma_corrected = gamma_corrected.astype(np.uint8)

    # 4️⃣ Contrast Stretching (Normalization)
    normalized = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # 5️⃣ Unsharp Masking (Sharpening)
    gaussian_blur = cv2.GaussianBlur(image, (9, 9), 10.0)
    unsharp_mask = cv2.addWeighted(image, 1.5, gaussian_blur, -0.5, 0)

    # 6️⃣ Laplacian Filtering for Edge Contrast
    laplacian_filtered = cv2.Laplacian(image, cv2.CV_64F)
    laplacian_filtered = cv2.convertScaleAbs(laplacian_filtered)

    # Store results
    results = {
        "Original": image,
        "Histogram Equalization": hist_equalized,
        "Adaptive Histogram Equalization (CLAHE)": adaptive_hist_equalized,
        "Gamma Correction": gamma_corrected,
        "Contrast Stretching (Normalization)": normalized,
        "Unsharp Masking (Sharpening)": unsharp_mask,
        "Laplacian Filtering": laplacian_filtered,
    }

    return results

# Define the image path
image_path = r'C:\Users\User\Documents\PythonProjects\E2CR\sample_images_for_ocr\pp_denoised.tiff'
image_path = "C:/Users/User/Documents/PythonProjects/E2CR/sample_images_for_ocr/4159363_00363.jpg"

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#contrast_results = apply_contrast_filters(image_path)

#image = contrast_results['Gamma Correction']

image = crop_to_border(image)
image = denoise(image)
image = sharpen(image)

image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)

cv2.imshow('FINAL IMAGE',image)
cv2.waitKey()

image[image != 255] = 0
image = denoise(image)
cv2.imwrite(r'C:\Users\User\Documents\PythonProjects\E2CR\debug\cany_extracted.tiff',image)
assert False





edge_image = cv2.Canny(image, 10, 100)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 # Generate random colors for each contour
colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in contours]

# Draw each contour in a different color
for i, contour in enumerate(contours):
    #cv2.drawContours(image, [contour], -1, colors[i], thickness=1)
    pass
cv2.imshow('image',image)




# Normalize the image to enhance contrast
image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

# Apply a light Gaussian blur to smooth gray values
blurred = cv2.GaussianBlur(image, (5, 5), 0)


# Create a mask of gray areas (between dark and white)
gray_mask = (image > 100) & (image < 230)

# Linearly push gray regions toward white
merged = image.astype(np.float32)  # Convert to float for smooth adjustments
merged[gray_mask] += (255 - merged[gray_mask]) * 1.0  # Gradually push gray to white
merged = np.clip(merged, 0, 255).astype(np.uint8)  # Clip and convert back

cv2.imwrite(r'C:\Users\User\Documents\PythonProjects\E2CR\debug\cany_extracted.tiff',merged)




cv2.imshow('sharpended',merged)
cv2.waitKey()

def edge_detection(image, method="sobel"):
    """
    Detects edges in an image using Sobel, Canny, or Laplacian method.
    
    Parameters:
    - image_path (str): Path to the input image.
    - method (str): The edge detection method. Options: "sobel", "canny", "laplacian".
    
    Returns:
    - edge_image (numpy.ndarray): The detected edge map.
    """
    
    # Load the image in grayscale
    
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

merged = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)

vcc = visualize_cc(cv2.bitwise_not(merged))
cv2.imshow("Connected Components",vcc)
cv2.waitKey()

# Plot all edge detection methods
plt.figure(figsize=(12, 4))
for i, method in enumerate(edge_methods):
    edges = edge_detection(merged, method)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank black image (same size as original)
    contour_image = np.ones_like(image)

    # Draw contours on the blank image
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), thickness=2)
    
    
        
    plt.subplot(1, 3, i + 1)
    plt.imshow(contour_image, cmap="gray")
    plt.title(f"{method.capitalize()} Edges")
    plt.axis("off")

plt.tight_layout()
plt.show()


