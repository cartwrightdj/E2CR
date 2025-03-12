import cv2
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import hilbert, find_peaks, savgol_filter, detrend
from scipy.ndimage import gaussian_filter1d
import heapq
from scipy.ndimage import convolve

image_path = r'C:\Users\User\Documents\PythonProjects\E2CR\sample_images_for_ocr\preprocessed1.tiff'
image_path = r'C:\Users\User\Documents\PythonProjects\E2CR\sample_images_for_ocr\cany_extracted.tiff'

#image_path = r'C:\Users\User\Documents\PythonProjects\E2CR\debug\cany_extracted.tiff_Image Pre-Processing Final Result.tiff'
image_path = r'C:\Users\User\Documents\PythonProjects\E2CR\sample_images_for_ocr\pp_at_fullpage.tiff'
image_path = r'C:\Users\User\Documents\PythonProjects\E2CR\debug\cany_extracted.tiff'

image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

height, width = image.shape
print(f"width: {width}, height:{height}")

data = np.sum(image, axis=1)

# Apply Savitzky-Golay filter to smooth signal
data = savgol_filter(data, window_length=15, polyorder=2)  # Smoother version

avg = np.mean(data)
std = np.std(data)

smoothed_signal = savgol_filter(data, window_length=31, polyorder=3)
plt.plot(data, range(len(data)), linestyle='-', color='b', label='Origional Data')
plt.plot(smoothed_signal, range(len(data)), linestyle='-', color='y', label='Origional Data Smoothed')
plt.show()

# Compute Hilbert transform envelope
analytic_signal = hilbert(data)
envelope = np.abs(analytic_signal)

# Apply additional Gaussian smoothing
envelope_smoothed = gaussian_filter1d(envelope, sigma=3)

# 🔥 Use Polynomial Trend Removal Instead of Detrend
x = np.arange(len(envelope_smoothed))  # X-axis (row indices)
p = np.polyfit(x, envelope_smoothed, deg=2)  # Fit a quadratic trend
trend = np.polyval(p, x)  # Compute the trend
deskewed_envelope = envelope_smoothed - trend  # Subtract trend to deskew

# Apply Detrending to Remove Skew (Deskewing)
deskewed_envelope = detrend(envelope_smoothed)
print(f"Minimum in Envolope: {np.min(deskewed_envelope)}")

# 🔄 Flip Around the Center Value (Mean of Row Sums)
#center_value = np.mean(data)  # Compute the center
deskewed_envelope = deskewed_envelope + int((avg-np.mean(deskewed_envelope))) # Reflect around the center


#peaks, _ = find_peaks(data, height=avg, prominence=(.5 * std))
peaks, _ = find_peaks(deskewed_envelope, height=np.mean(deskewed_envelope), prominence=(.75 * np.std(deskewed_envelope)))
print(f"Found {len(peaks)} in deskewed_envelope")

plt.plot(data, range(len(data)), linestyle='-', color='b', label='Data')
#plt.plot(envelope_smoothed, range(len(data)), linestyle='-', color='y', label='Envelope')  # Hilbert envelope
plt.plot(deskewed_envelope, range(len(data)), linestyle='--', color='r', label='Envelope')  # Hilbert envelope

plt.axvline(avg,linestyle=':',color='g')
plt.axvline(avg-std,linestyle=':')

plt.axvline(np.mean(deskewed_envelope),linestyle=':',color='g')
plt.axvline(np.mean(deskewed_envelope)-np.std(deskewed_envelope),linestyle=':',color='g')


plt.gca().invert_yaxis() 
for peak in peaks:
    #cv2.line(image, (0, peak), (width, peak), (0, 255, 0), thickness=1)
    plt.axhline(peak,linestyle=':',color='green')
#plt.axvline(x=threshold, color='r', linestyle='--', linewidth=1, label=f'Threshold ({threshold})')
#plt.scatter(np.array(data)[peaks], peaks, color='green', s=100, zorder=3, label='Peaks above threshold')

#for peak in peaks:
#    plt.axhline(y=peak, color='g', linestyle=':', linewidth=1)
#    plt.annotate(f"{peak}", (data[peak], peak), textcoords="offset points", xytext=(0,10), ha='center', color='green')
    
plt.xlabel('Normalized Value')
plt.ylabel('Row Index')
plt.show()

cv2.imwrite(r'C:\Users\User\Documents\PythonProjects\E2CR\debug\sci_peaks.jpg',image)


#############################################################################################################################

import cv2
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import matplotlib.pyplot as plt

def image_to_graph(image, start_y, max_deviation=20, white_penalty_factor=10000):
    """
    Converts an image into a sparse graph where:
    - Darker pixels (near 0) have very low costs (preferred paths).
    - White pixels (near 65535) are **heavily penalized** to prevent crossing.
    - Movement is restricted to within `max_deviation` pixels from `start_y`.

    Arguments:
    - image: np.ndarray, grayscale image in uint16 (0-65535).
    - start_y: int, initial row index.
    - max_deviation: int, max allowed vertical deviation from start_y.
    - white_penalty_factor: int, penalty multiplier for white pixels.

    Returns:
    - sparse_graph: scipy.sparse matrix representing the graph.
    - shape: Tuple (height, width) for path reconstruction.
    """
    height, width = image.shape
    num_pixels = height * width

    # Normalize cost: Scale 0-65535 to 0-1 range (low values = low cost, high values = high cost)
    cost_map = image.astype(np.float64) / 65535.0

    # Strongly penalize white pixels
    cost_map = np.where(image > 25000, cost_map * white_penalty_factor, cost_map)

    # Set completely black pixels (`0`) to **almost zero cost** (ensuring they are chosen)
    cost_map = np.where(image == 0, 0.00001, cost_map)

    # Store edges for sparse matrix
    row_idx = []
    col_idx = []
    weights = []

    max_cost = np.max(cost_map)
    for y in range(max(0, start_y - max_deviation), min(height, start_y + max_deviation + 1)):  # Uses start_y
        for x in range(width):
            index = y * width + x

            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, Down, Left, Right
                ny, nx = y + dy, x + dx

                # Restrict vertical movement within max_deviation
                if not (start_y - max_deviation <= ny <= start_y + max_deviation):
                    continue

                if 0 <= ny < height and 0 <= nx < width:
                    neighbor_index = ny * width + nx
                    weight = max(cost_map[y, x], cost_map[ny, nx])  # Use max to enforce stronger penalties
                    if y < start_y:
                        delta = abs(y - start_y) 
                        r = 100 / max_deviation / 100
                        weight = weight + (max_cost * (r * delta) * .15)
                    if y > start_y:
                        delta = abs(y - start_y) 
                        r = 100 / max_deviation / 100
                        weight = weight + (max_cost * (r * delta) * .05)
                    row_idx.append(index)
                    col_idx.append(neighbor_index)
                    weights.append(weight)

    # Convert to sparse matrix
    sparse_graph = scipy.sparse.csr_matrix((weights, (row_idx, col_idx)), shape=(num_pixels, num_pixels))

    return sparse_graph, (height, width)

def find_shortest_path(image, start, end, max_deviation=20, white_penalty_factor=10000):
    """
    Finds the shortest path across an image using Dijkstra’s algorithm.

    Arguments:
    - image: np.ndarray, grayscale image.
    - start: Tuple (y, x) as the starting pixel.
    - end: Tuple (y, x) as the target pixel.
    - max_deviation: int, max allowed deviation from start_y.
    - white_penalty_factor: int, penalty multiplier for white pixels.

    Returns:
    - path: List of (y, x) coordinates representing the shortest path.
    """
    start_y = start[0]
    sparse_graph, (height, width) = image_to_graph(image, start_y, max_deviation, white_penalty_factor)

    # Convert (y, x) to graph indices
    start_index = start[0] * width + start[1]
    end_index = end[0] * width + end[1]

    # Compute shortest paths from start pixel
    distances, predecessors = scipy.sparse.csgraph.dijkstra(sparse_graph, directed=True, indices=start_index, return_predecessors=True)

    # If no valid path exists
    if distances[end_index] == np.inf:
        print("No valid path found.")
        return []

    # Backtrack to construct the path
    path_indices = []
    current = end_index
    while current != -9999:  # -9999 means no predecessor
        path_indices.append(current)
        current = predecessors[current]

    # Convert node indices back to (y, x) coordinates
    path_coords = [(idx // width, idx % width) for idx in path_indices]

    return path_coords[::-1]  # Reverse to get the correct order

def visualize_path(image, paths):
    """
    Overlays the found path on the image for visualization.

    Arguments:
    - image: np.ndarray, grayscale image.
    - path: List of (y, x) coordinates representing the path.
    """
    image = image.astype(np.uint8)
    image = np.bitwise_not(image)
    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    for path in paths:
        for y, x in path:
            img_color[y, x] = (0, 255, 0)  # Mark path in green

    cv2.imwrite(r'C:\Users\User\Documents\PythonProjects\E2CR\debug\paths.tiff',img_color)
    plt.figure(figsize=(10, 6))
    plt.imshow(img_color, cmap='gray')
    plt.title("Shortest Path Overlaid on Image")
    plt.axis("off")
    plt.show()

# Load grayscale image
#image_path = r'C:\Users\User\Documents\PythonProjects\E2CR\debug\shaded.tiff'
image_path = r'C:\Users\User\Documents\PythonProjects\E2CR\debug\cany_extracted.tiff'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image_to_crop = image.copy()
image = np.bitwise_not(image)


# Ensure image is loaded correctly
if image is None:
    raise ValueError("Error: Image could not be loaded. Check the file path!")

# Convert image to uint16 if necessary
if image.dtype != np.uint16:
    image = image.astype(np.uint16)

paths = []
for peak in peaks:
    # Define start and end points
    start = (peak, 0)  # Start from row 50, leftmost column
    end = (peak, image.shape[1] - 1)  # End at row 50, rightmost column

    # Set max deviation constraint
    max_deviation = 50  # Path cannot deviate more than ±20 pixels from start_y

    # Find shortest path
    path = find_shortest_path(image, start, end, max_deviation, white_penalty_factor=10000)
    paths.append(path)

# Visualize result
#visualize_path(image, paths)

from e2cr import cropSegmentFromImage


c_images = cropSegmentFromImage(image_to_crop,paths)
print(f"{len(c_images)} rows found")
for i, c_image in enumerate(c_images):
    print(f"saving row {i+1}")
    cv2.imwrite(f'C:/Users/User/Documents/PythonProjects/E2CR/segmentation/row_{i}.tiff',c_image)







