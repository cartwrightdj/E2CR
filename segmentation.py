import cv2
import numpy as np
from plotting import *
from scipy.signal import find_peaks
#from common import *
from imageman_manipulation import *
from utils import *
from sklearn.cluster import DBSCAN
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from loguru import logger
from heapq import heappop, heappush

def getImageHist(ImageToHist, axis, exclude_edges=0):
    """
    Calculate the histogram or projection of an image across a given axis, excluding a certain number of pixels from the edges.

    Arguments:
    ImageToHist -- np.ndarray, the input image
    axis -- int, the axis along which to calculate the projection (0 for vertical, 1 for horizontal)
    exclude_edges -- int, the number of pixels to exclude from the edges

    Returns:
    projection -- np.ndarray, the projection of the image along the specified axis
    """
    # Add check here for a valid axis

    # Check if the input image is grayscale; if not, convert it to grayscale
    if len(ImageToHist.shape) == 3:
        gray_image = cv2.cvtColor(ImageToHist, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = ImageToHist

    # Exclude the specified number of pixels from the edges
    if axis == 0:
        cropped_image = gray_image[exclude_edges:gray_image.shape[0] - exclude_edges, :]
    else:
        cropped_image = gray_image[:, exclude_edges:gray_image.shape[1] - exclude_edges]

    # Calculate the projection by summing pixel values along the specified axis

    #cv2.imwrite(f"E:/E2CR_NM/output/projection)image.jpg", ImageToHist)
    projection = np.sum(cropped_image, axis=axis)

    ''' Plot the horizontal projection
    plt.figure(figsize=(10, 5))
    plt.title('Horizontal Projection (Row-wise Histogram)')
    plt.plot(projection)
    plt.xlabel('Row Index')
    plt.ylabel('Sum of Pixel Values')
    plt.show()
    '''

    return projection

def find_high_points(data, threshold, distance=None, prominence=None) -> np.ndarray:
    #print(f"{Colors.OKBLUE}\nE2CR Segmentation: Locating Maxima of {len(data)} points above {round(threshold,2)}{Colors.ENDC}")
    #Xprint(f"{Colors.OKBLUE}\tthreshold={threshold}, distance={distance}, prominence={prominence}{Colors.ENDC}")
    """
    Find high points (peaks) in the data that are above the specified threshold.
    
    Args:
        data (list or np.array): The input data array.
        threshold (float): The threshold value.
        distance (int, optional): Minimum horizontal distance (in samples) between neighboring peaks.
        prominence (float, optional): Required prominence of peaks.
    
    Returns:
        list: Indices of high points (peaks) above the threshold.
    """
    peaks, _ = find_peaks(data, height=threshold, distance=distance, prominence=prominence)
    hp_loss = (1 - len(peaks) / len(data)) * 100
    #print(f"{Colors.OKBLUE}E2CR Segmentation: Returning {len(peaks)} of {len(data)} points, loss: {hp_loss}%{Colors.ENDC}")
    return peaks 

def group_by_proximity(data, eps, min_samples):
    """
    Group numbers in a list by their proximity to each other using DBSCAN clustering.
    
    Args:
        data (list or np.array): List of numerical data.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    
    Returns:
        tuple: A tuple containing a list of lists (clusters) and the labels for each point.

        eps (float): defines the maximum distance between two samples for one to be considered as in the neighborhood of the other. 
        It is essentially the radius of the neighborhood around a point.
            Smaller eps: Decreases the size of the neighborhood, which can lead to more clusters because fewer points will be within each other’s neighborhood.
            Larger eps: Increases the size of the neighborhood, which can lead to fewer clusters because more points will be within each other’s neighborhood, potentially merging clusters.

        min_samples: Description: min_samples is the minimum number of points required in a neighborhood to form a core point. A core point is a point that has at least min_samples points (including itself) within its eps neighborhood.
            Effect on Clustering:
            Smaller min_samples: Reduces the number of points required to form a dense region, which can result in more clusters since even smaller groups can form a cluster.
            Larger min_samples: Increases the number of points required to form a dense region, which can lead to fewer clusters since only larger groups can form a cluster.
    """
    if len(data) < 1: 
        logger.warning(f"Segmentation: Grouping by Proximity of {len(data)} break points; eps={eps}, min_samples={min_samples}")
        return [], np.ndarray(1)
    
    logger.debug(f"Received {len(data)} break points; eps={eps}, min_samples={min_samples}")
    logger.debug(f"Break Points: {data}")

    # Convert data to numpy array and reshape for DBSCAN
    data = np.array(data).reshape(-1, 1)
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    labels = clustering.labels_
    
    # Initialize clusters list
    clusters = []
    # Iterate over unique labels to form clusters
    for label in set(labels):
        if label != -1:  # Exclude noise points labeled as -1
            cluster = data[labels == label].flatten().tolist()
            clusters.append(cluster)
    
    # Calculate and log effective loss
    chp_loss = round((1 - len(clusters) / len(data)) * 100, 2)
    logger.debug(f"Returning {len(clusters)} clusters; eps={eps}, min_samples={min_samples}, effective loss={chp_loss}%.")
    logger.debug(f"Clusters: {clusters}")
    
    return clusters, labels

def find_breakpoints_max(clusters, imageHist):
    """
    Find the breakpoints in the clusters based on the highest value in imageHist coresponding to the indexes in the cluster.
    
    Args:
        clusters (list): List of clusters.
        imageHist (list or np.array): List of histogram values corresponding to the data indices.
    
    Returns:
        list: A list of breakpoints.
    """
    logger.debug(f"recieved {len(clusters)}")
    logger.trace(f"Cluster(s): {clusters}")

    BreakPoints = []
    BreakValues = []
    for cluster in clusters:
        max_value = -np.inf
        index = 0
        for i, value in enumerate(cluster):
            if imageHist[value] > max_value:
                max_value = imageHist[value]
            index = i
        BreakPoints.append(cluster[index])
        BreakValues.append(imageHist[cluster[index]])

    logger.debug(f"Found Max in {len(clusters)} Clusters, returning {len(BreakPoints)} break points.")
    logger.trace(f"{len(BreakPoints)} Break Points: {BreakPoints}")
    return BreakPoints, BreakValues

def find_breakpoints_average(clusters,y_data):
    """
    Find the x value in each cluster that is closest to the average of all the y values for that cluster.
    
    Args:
        y_data (list or np.array): List of y values corresponding to the data indices.
        clusters (list): List of clusters.
    
    Returns:
        list: A list of breakpoints.
    """
    def split_large_breaks(int_list, threshold):
        # Initialize a list to hold the additional elements
        additions = []

        # Iterate through the list and check the distances
        int_list.insert(0,0)
        for i in range(len(int_list) - 1):
            diff = abs(int_list[i + 1] - int_list[i])
            if diff > threshold:
                avg = int(round((int_list[i + 1] + int_list[i]) / 2,2))
                print(f"Breaking {int_list[i + 1]} and {int_list[i]}")
                additions.append(avg)

        # Extend the original list with the new elements
        int_list.extend(additions)

        # Sort the list
        sorted_list = sorted(int_list)
        return sorted_list
    
    breakpoints = []
     
    for cluster in clusters:
        y_values = [y_data[value] for value in cluster]
        average_y = np.mean(y_values)
        closest_value = min(cluster, key=lambda x: abs(y_data[x] - average_y))
        breakpoints.append(closest_value)

    breakpoints = split_large_breaks(breakpoints, SEG_DEFAULT.EXPECTED_IMAGE_HW * 2)
    

    differences = [breakpoints[i+1] - breakpoints[i] for i in range(len(breakpoints) - 1)]
    '''
    mean_diff = np.mean(differences)  
    median_diff = np.median(differences)   
    mode_diff = stats.mode(differences)[0]  # mode() returns a mode array
    print(f"{Colors.OKBLUE}E2CR Segmentation: Average Break Points Statistics")
    print("\tDifferences:", differences)
    print("\tMean:", mean_diff)
    print("\tMedian:", median_diff)
    print("\tMode:", mode_diff, f"\n{Colors.ENDC}")   
    '''

    return breakpoints
    
def find_shortest_path(image, gradients, y, log_cost_factor=15, bias_factor=10, gradient_factor=5):
    """
    Find the shortest path in a binary image using a modified Dijkstra's algorithm with exponential and bias costs.

    Arguments:
    image -- np.ndarray, the binary image
    gradients -- np.ndarray, the gradient magnitudes of the image
    y -- int, the starting row
    log_cost_factor -- float, the factor for logarithmic cost. (Penalizes vertical movements based on the exponential distance.
    bias_factor -- float, the factor for bias cost (Penalizes deviations from the starting row.)
    gradient_factor -- float, the factor for gradient cost (Penalizes crossing high-gradient regions, discouraging the path from crossing text lines or words.)
    max_exp_cost -- float, the maximum allowable exponential cost to prevent overflow

    Returns:
    path -- list of tuples, the sequence of points in the shortest path
    max_y -- int, the maximum y-coordinate in the path
    min_y -- int, the minimum y-coordinate in the path

    log_cost_factor -- float, the factor for logarithmic cost. (Penalizes vertical movements based on the exponential distance.)
        This parameter scales the logarithmic penalty. It determines how strongly the pathfinding algorithm discourages vertical movements. 
        A higher log_cost_factor means that even small vertical deviations will incur a significant penalty, leading to a preference for straighter paths.
        
    bias_factor -- float, the factor for bias cost (Penalizes deviations from the starting row.)

        The bias_factor is used to add a cost based on the vertical deviation from the starting row y. 
        Specifically, it penalizes movements that result in a significant change in the y-coordinate. 
        This helps to control how much the path can deviate vertically.

        Low bias_factor (e.g., bias_factor = 1):
            Effect: The pathfinding algorithm will allow more vertical deviations because the penalty for changing the y-coordinate is relatively small.
            Behavior: The path may zigzag more and deviate significantly from the starting row if doing so leads to a lower cumulative path cost.

        High bias_factor (e.g., bias_factor = 100):
            Effect: The pathfinding algorithm will strongly discourage vertical deviations because the penalty for changing the y-coordinate is high.
            Behavior: The path will tend to stay close to the starting row and will only deviate vertically if absolutely necessary. This results in a straighter path.
    """
    logger.debug(f"Finding Shortest Path from: {y}")
    logger.trace(f"y: {y}, log_cost_factor={log_cost_factor}, bias_factor={bias_factor}, gradient_factor={gradient_factor}")
    
    rows, cols = image.shape
    dist = np.full((rows, cols), np.inf, dtype=np.float64)
    prev = np.full((rows, cols), None, dtype=object)
    dist[y, 0] = float(image[y, 0])
    queue = [(float(image[y, 0]), y, 0)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        curr_dist, cy, cx = heappop(queue)
        if cx == cols - 1:
            break
        for dy, dx in directions:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < rows and 0 <= nx < cols:
                log_cost = log_cost_factor * np.log1p(abs(ny - cy))
                bias_cost = bias_factor * abs(ny - y)
                grad_cost = gradient_factor * gradients[ny, nx]
                new_dist = curr_dist + float(image[ny, nx]) + log_cost + bias_cost + grad_cost
                if new_dist < dist[ny, nx]:
                    dist[ny, nx] = new_dist
                    prev[ny, nx] = (cy, cx)
                    heappush(queue, (new_dist, ny, nx))

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

def calculate_gradients(image):
    """
    Calculate the gradient magnitudes of the image.

    Arguments:
    image -- np.ndarray, the input binary image

    Returns:
    gradients -- np.ndarray, the gradient magnitudes of the image
    """
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradients = cv2.magnitude(grad_x, grad_y)

    '''# Plot the gradients
    plt.figure(figsize=(10, 5))
    plt.title('Gradient Magnitudes')
    plt.imshow(gradients, cmap='gray')
    plt.colorbar()
    plt.show()
    '''
    
    return gradients

def getLineStats(line):
    """
    Compute the minimum and maximum y-coordinates for a given line.

    Args:
        line (list of tuples): A line represented by a list of (y, x) points.

    Returns:
        tuple: Minimum and maximum y-coordinates of the line.
    """
    if not line:
        raise ValueError("Line is empty")

    min_y = min(y for y, x in line)
    max_y = max(y for y, x in line)

    return min_y, max_y
