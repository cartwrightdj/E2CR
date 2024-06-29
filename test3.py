import cv2
import numpy as np

def calculate_and_draw_section_stats(image, n):
    """
    Break an image into n horizontal parts and draw statistics for each section on the image,
    including the percent of non-white pixels.

    Args:
        image (np.ndarray): The input image.
        n (int): The number of horizontal parts to divide the image into.
    """
    if not is_grayscale(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = image.shape
    section_height = height // n
    stats_list = []

    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for i in range(n):
        start_row = i * section_height
        end_row = (i + 1) * section_height if i < n - 1 else height
        section = image[start_row:end_row, :]

        # Calculate statistics
        mean = np.mean(section)
        median = np.median(section)
        std_dev = np.std(section)
        non_white_pixels = np.sum(section < 245)
        total_pixels = section.size
        non_white_percent = (non_white_pixels / total_pixels) * 100

        # Prepare text
        text = (f"Mean: {mean:.2f}\t"
                f"Median: {median:.2f}\t"
                f"total_pixels: {total_pixels:.2f}\n"
                f"Std Dev: {std_dev:.2f}\n"
                f"Non-white Pixels: {non_white_pixels}\n"
                f"Percent Non-white Pixels: {non_white_percent:.2f}%")
        
        # Draw text on the image
        y0, dy = start_row + 20, 15
        for j, line in enumerate(text.split('\n')):
            y = y0 + j * dy
            cv2.putText(color_image, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        
        # Draw a line at the bottom of each section
        if i < n - 1:
            cv2.line(color_image, (0, end_row), (width, end_row), (0, 255, 0), 2)

        stats_list.append({
            "mean": mean,
            "median": median,
            "std_dev": std_dev,
            "non_white_pixels": non_white_pixels,
            "non_white_percent": non_white_percent
        })

    cv2.imwrite('output_with_stats.jpg', color_image)
    return color_image, stats_list

def is_grayscale(image):
    """
    Check if an image is grayscale.

    Args:
        image (np.ndarray): The input image.

    Returns:
        bool: True if the image is grayscale, False otherwise.
    """
    return len(image.shape) == 2 or image.shape[2] == 1



# Example usage
image_path = "./sample_images_for_ocr/4159363_00370.jpg"
image = cv2.imread(image_path)
n = 23  # Number of horizontal parts
output_image, stats_list = calculate_and_draw_section_stats(image, n)
cv2.imshow("Image with Stats", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
