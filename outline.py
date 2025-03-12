import cv2
import numpy as np

def outline_black_pixels(image, edge_thickness=1):
    """
    Creates an outline around black pixels in a grayscale image.

    Arguments:
    - image: np.ndarray, grayscale image (0-255).
    - edge_thickness: int, thickness of the outline.

    Returns:
    - outlined_image: np.ndarray, image with black pixels outlined in white.
    """
    # Ensure the image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(image, threshold1=10, threshold2=50)  # Detects black pixel edges

    # Convert edges to white on a black background
    outlined_image = np.zeros_like(image)
    outlined_image[edges > 0] = 255  # Set edges to white

    # Optionally thicken the edges
    if edge_thickness > 1:
        kernel = np.ones((edge_thickness, edge_thickness), np.uint8)
        outlined_image = cv2.dilate(outlined_image, kernel, iterations=1)

    return outlined_image

# Load grayscale image
image_path = r'C:\Users\User\Documents\PythonProjects\E2CR\debug\cany_extracted.tiff'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply black pixel outlining
outlined_image = outline_black_pixels(image, edge_thickness=2)

# Display result
cv2.imshow("Outlined Black Pixels", outlined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
