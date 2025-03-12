import cv2
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def extract_txt_by_contour_mask(image):
    # image: BLACK on WHITE image
    image = cv2.bitwise_not(image)

    edge_image = cv2.Canny(image, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edge_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f'Found {len(contours)} contours.')

    # Create a blank black image (same size as original)
    contour_image = np.zeros_like(image)
    
    # Draw contours on the blank image
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), thickness=3)
   

    #image = cv2.bitwise_and(image,black_bg,mask=contour_image)
    white_background = np.ones_like(image) * 255

    image = cv2.bitwise_not(image)
    # Extract the portion of the image under the mask
    extracted = cv2.bitwise_and(image, image, mask=contour_image)
    
    # Copy extracted region onto the white background
    white_background[contour_image == 255] = image[contour_image == 255]
    #white_background[white_background != 255] = 0
    return white_background

def extract_txt_by_contour_mask_sobel(image):

    image = cv2.bitwise_not(image)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in X direction
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in Y direction
    edge_image = np.sqrt(sobelx**2 + sobely**2)  # Compute gradient magnitude
    edge_image = (edge_image / edge_image.max() * 255).astype(np.uint8)  # Normalize to 0-255

    cv2.imshow("edge_image",edge_image)
    cv2.waitKey()

    # Apply thresholding to convert edges into binary format
    _, binary_edges = cv2.threshold(edge_image, 15, 255, cv2.THRESH_BINARY)

    # Apply morphological closing to merge broken edges
    kernel = np.ones((3, 3), np.uint8)
    closed_edges = cv2.morphologyEx(binary_edges, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("sobel edge_image",closed_edges)
    cv2.waitKey()

    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours.")

    # Create a blank black image (same size as original)
    contour_image = np.zeros_like(image)

        
    # Draw contours on the blank image
    #cv2.drawContours(contour_image, contours[2:], -1, (0, 0, 0), thickness=cv2.FILLED)
    cv2.drawContours(contour_image, contours[2:], -1, (255, 255, 255), thickness=1)

    cv2.imshow("ci",contour_image)
    cv2.waitKey()

    #image = cv2.bitwise_and(image,black_bg,mask=contour_image)
    white_background = np.ones_like(image) * int(image_stats(image)['Mean Intensity'])
    print(f"image_stats(image)['Mean Intensity']: {image_stats(image)['Mean Intensity']}")

    image = cv2.bitwise_not(image)
    # Extract the portion of the image under the mask
    extracted = cv2.bitwise_and(image, image, mask=contour_image)

    # Copy extracted region onto the white background
    white_background[contour_image == 255] = extracted[contour_image == 255]
    white_background[white_background < 100] = 0
    return white_background

def image_stats(image):
    """
    Computes comprehensive image statistics including:
    - Basic Stats: Mean, Std Dev, Min/Max values, Histogram
    - Advanced Stats: Entropy, Skewness, Kurtosis, RMS Contrast
    - Structural Stats: Contour count, Bounding Box, Aspect Ratio, Extent, Solidity
    
    Parameters:
    - image_path (str): Path to the input image.
    
    Returns:
    - Dictionary with all computed statistics.
    """
    # Load image in grayscale
    
    
    # ------------------------- 1️⃣ Basic Stats -------------------------
    mean, stddev = cv2.meanStdDev(image)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(image)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    # ------------------------- 2️⃣ Advanced Stats -------------------------
    entropy = stats.entropy(hist.flatten() + 1e-10)  # Avoid log(0)
    skewness = stats.skew(image.flatten())
    kurtosis = stats.kurtosis(image.flatten())
    contrast_rms = np.std(image)  # RMS Contrast
    
    
    
    # ------------------------- Return All Stats -------------------------
    return {
        "Mean Intensity": mean[0][0],
        "Standard Deviation": stddev[0][0],
        "Min Pixel Value": min_val,
        "Max Pixel Value": max_val,
        "Min Pixel Location": min_loc,
        "Max Pixel Location": max_loc,
        "Entropy": entropy,
        "Skewness": skewness,
        "Kurtosis": kurtosis,
        "Contrast (RMS)": contrast_rms,
        "Histogram": hist
    }



'''
Common Pre-Processing Steps

1. Crop out any border in the image as it will negativly affect contour and edge detection as well as historgrams
'''
def crop_to_border(image,threshold=0.5):
    """
    Crop the image to the border where significant content exists.

    Args:
        image (np.array): The input image.

    Returns:
        np.array: The cropped image containing only the significant content.
    """
    # Ensure the image is not empty
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or invalid")

    # Check if the input is a color image and keep the original
    is_color = len(image.shape) == 3
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if is_color else image

    # Apply thresholding to segment handwriting
    _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    # Calculate the sum of pixels for each row and column
    row_sums = np.sum(binary, axis=1)
    col_sums = np.sum(binary, axis=0)

    # Find the rows and columns where the sums exceed a threshold
    row_threshold = np.mean(row_sums) * threshold
    col_threshold = np.mean(col_sums) * threshold

    rows_with_content = np.where(row_sums > row_threshold)[0]
    cols_with_content = np.where(col_sums > col_threshold)[0]

    if len(rows_with_content) == 0 or len(cols_with_content) == 0:
        return image  # No significant content found, return the original image

    # Determine the bounding box of the content
    y_min, y_max = rows_with_content[0], rows_with_content[-1]
    x_min, x_max = cols_with_content[0], cols_with_content[-1]

    # Crop the image to the bounding box
    cropped_image = image[y_min:y_max, x_min:x_max]

    return cropped_image

def clean_image(image):
    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(image, (5, 5), 1.5)

    # Unsharp masking: Original + (Original - Blurred) * weight
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    denoised = cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)

def denoise(image=None, templateWindowSize=7, searchWindowSize=21):
    denoised_image = cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)
    return denoised_image

def sharpen(image, kernel=(5,5),sigma=1.5, α=1.5,β=-0.5, γ=0):
    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(image, kernel, sigma)

    # Unsharp masking: Original + (Original - Blurred) * weight
    sharpened = cv2.addWeighted(image, α, blurred, β, γ)
    return sharpened

def remove_gray_halo(image):
    dvals = image[image !=255]
    dvals = dvals[dvals !=0]
    davg = np.mean(dvals)
    dstd = np.std(dvals)
    print(davg)
    image[image >= davg+(.75 * dstd)] = 255
    image[image < davg+(.75 * dstd)] = 0
    print(f'davg: {davg}')

    unique_values, counts = np.unique(dvals, return_counts=True)

    # Plot bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(unique_values, counts, color='blue', alpha=0.7, edgecolor='black')
    plt.axvline(davg,linestyle=':')
    plt.axvline(davg+(.5 * dstd),linestyle=':')
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.title("")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

