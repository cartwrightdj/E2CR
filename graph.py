import cv2
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import matplotlib.pyplot as plt

def create_sparse_graph(image, white_cost=10):
    """
    Converts a black & white image into a sparse graph where:
    - Black pixels (0) have zero cost.
    - White pixels (255) have a cost defined by `white_cost`.
    - Each pixel is connected to its 4 neighbors (up, down, left, right).

    Returns:
    - sparse_graph: Sparse adjacency matrix (CSR format)
    - pixel_indices: Dictionary mapping (y, x) coordinates to graph indices
    - shape: Tuple (height, width) for reconstruction
    """

    height, width = image.shape
    num_pixels = height * width  # Total number of pixels
    pixel_indices = {}  # Store mapping from (y, x) → graph index
    edges = []
    weights = []

    # Convert image to binary (0 = black, 255 = white)
    binary_image = np.where(image > 128, 255, 0).astype(np.uint8)

    # Create graph by connecting each pixel to its 4 neighbors
    for y in range(height):
        for x in range(width):
            index = y * width + x
            pixel_indices[(y, x)] = index

            # Determine pixel cost
            pixel_cost = 0 if binary_image[y, x] == 0 else white_cost

            # Connect to 4-neighbors (up, down, left, right)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    neighbor_index = ny * width + nx
                    neighbor_cost = 0 if binary_image[ny, nx] == 0 else white_cost
                    edges.append((index, neighbor_index))
                    weights.append(neighbor_cost)

    # Convert edge list to sparse adjacency matrix
    row_idx, col_idx = zip(*edges)
    sparse_graph = scipy.sparse.csr_matrix((weights, (row_idx, col_idx)), shape=(num_pixels, num_pixels))

    return sparse_graph, pixel_indices, (height, width)

def find_shortest_path(image, start_y, white_cost=10):
    """
    Finds the shortest path from a given y-position on the leftmost column
    to the rightmost column using Dijkstra's algorithm.

    Arguments:
    - image: np.ndarray, binary image (0 or 255)
    - start_y: int, the y-coordinate to start from
    - white_cost: int, cost of traversing white pixels

    Returns:
    - path: List of (y, x) coordinates representing the shortest path
    """
    sparse_graph, pixel_indices, (height, width) = create_sparse_graph(image, white_cost)

    # Define start and end nodes
    start_node = pixel_indices[(start_y, 0)]
    end_nodes = [pixel_indices[(y, width - 1)] for y in range(height)]

    # Compute shortest paths from the starting pixel
    distances, predecessors = scipy.sparse.csgraph.dijkstra(
        sparse_graph, directed=False, indices=start_node, return_predecessors=True
    )

    # Find the shortest path to any end node
    best_end_node = min(end_nodes, key=lambda node: distances[node])

    # Backtrack to construct the shortest path
    path_indices = []
    current = best_end_node
    while current != -9999:  # -9999 means no predecessor
        path_indices.append(current)
        current = predecessors[current]

    # Convert node indices back to (y, x) coordinates
    path_coords = [(idx // width, idx % width) for idx in path_indices]

    return path_coords

def visualize_path(image, path):
    """
    Overlays the shortest path onto the original image and displays it.

    Arguments:
    - image: np.ndarray, the original grayscale image
    - path: List of (y, x) coordinates representing the shortest path
    """
    if not path:
        print("No valid path found!")
        return

    # Convert image to BGR for visualization
    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw path in red
    for y, x in path:
        img_color[y, x] = (0, 0, 255)  # Red path

    # Show result
    plt.figure(figsize=(8, 6))
    plt.imshow(img_color)
    plt.title("Shortest Path Overlaid on Image")
    plt.axis("off")
    plt.show()

# Load binary image
image_path = r'C:\Users\User\Documents\PythonProjects\E2CR\debug\cany_extracted.tiff'  # Replace with your image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Ensure binary format
image = np.bitwise_not(image)

# Set parameters
start_y = 50  # Example: Start from row index 50
white_cost = 10  # Cost of moving through white pixels

# Find shortest path
path = find_shortest_path(image, start_y, white_cost)

# Visualize result
visualize_path(image, path)
