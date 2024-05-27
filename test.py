image_path = 'E:\\E2CR\\sample_images_for_ocr\\onetime\\R. 317 (3).jpg'

import cv2
import numpy as np
from heapq import heappop, heappush
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Convert image to binary using adaptive threshold
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    return image, thresh

def find_shortest_path(image, y, exp_cost_factor=10):
    rows, cols = image.shape
    dist = np.full((rows, cols), np.inf)
    prev = np.full((rows, cols), None, dtype=object)
    dist[y, 0] = image[y, 0]
    queue = [(image[y, 0], y, 0)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    steps = 0
    while queue:
        curr_dist, cy, cx = heappop(queue)
        if cx == cols - 1:
            break
        for dy, dx in directions:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < rows and 0 <= nx < cols:
                # Exponential cost for y movement
                exp_cost = exp_cost_factor * np.exp(abs(ny - cy))
                new_dist = curr_dist + image[ny, nx] + exp_cost
                if new_dist < dist[ny, nx]:
                    dist[ny, nx] = new_dist
                    prev[ny, nx] = (cy, cx)
                    heappush(queue, (new_dist, ny, nx))
        steps += 1
        #if steps % 10000 == 0:
        #    print(f"Step {steps}: Processing column {cx}, row {cy}")

    path = []
    min_dist = np.inf
    min_row = -1
    for row in range(rows):
        if dist[row, cols - 1] < min_dist:
            min_dist = dist[row, cols - 1]
            min_row = row

    if min_row == -1:
        return path, None, None

    cy, cx = min_row, cols - 1
    max_y = min_y = cy
    while (cy, cx) != (y, 0):
        path.append((cy, cx))
        cy, cx = prev[cy, cx]
        if cy > max_y:
            max_y = cy
        if cy < min_y:
            min_y = cy
    path.append((y, 0))
    path.reverse()
    return path, max_y, min_y

def draw_path_on_image(output_image, path, max_y, min_y):
    if len(output_image.shape) == 2:  # If the image is grayscale
        return cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
    
    for (cy, cx) in path:
        output_image[cy, cx] = [0, 0, 255]

    # Annotate max and min y-values
    if path:
        cv2.putText(output_image, f"Max Y: {max_y}", (10, max_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(output_image, f"Min Y: {min_y}", (10, min_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return output_image


#image_path = "path_to_your_image/R. 317 (3).jpg"
thresholded_image_path = "path_to_save_thresholded_image/thresholded_image.jpg"
output_path = "path_to_save_output_image/shortest_path_output_final.jpg"

# Processing
original_image, thresh_image = load_and_preprocess_image(image_path)

output_image = original_image
for y in range(850,900,25):
    shortest_path, max_y, min_y = find_shortest_path(thresh_image, y, 12)
    output_image = draw_path_on_image(output_image, shortest_path, max_y, min_y)

    # Save images
cv2.imwrite(f'E:\E2CR\output\thresh_{y}.jpg', thresh_image)
cv2.imwrite(f'E:\E2CR\output\output_{y}.jpg', output_image)


# Optional: Display the images using matplotlib (for visualization)
'''
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title('Thresholded Image')
plt.imshow(thresh_image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Shortest Path on Original Image')
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.show()
'''











