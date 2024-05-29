import cv2
import numpy as np
from scipy.signal import find_peaks
from common import *
from sklearn.cluster import DBSCAN
from loguru import logger
from heapq import heappop, heappush
import matplotlib.pyplot as plt


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

def findInitialBreakPoints(image: np.ndarray, axis: str, thresholdRate: float, distance: int = None, prominence: float = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the histogram or projection of an image across a given axis and find high points (peaks) in the histogram.

    Args:
        image (np.ndarray): The input image.
        axis (str): The axis along which to calculate the projection ('x' for vertical, 'y' for horizontal).
        thresholdRate (float): The threshold value as a percentage of the maximum value to detect peaks.
        distance (int, optional): Minimum horizontal distance (in samples) between neighboring peaks.
        prominence (float, optional): Required prominence of peaks.

    Returns:
        peaks (np.ndarray): Indices of high points (peaks) above the threshold.
        projection (np.ndarray): The projection of the image along the specified axis.
    """
    logger.debug(f"Creating Image Projection along {axis} axis")

    # Check if the input axis is valid
    if axis not in ['x', 'y']:
        logger.critical(f"Improper axis passed: {axis}")
        raise ValueError(f"Improper axis passed: {axis}")

    # Check if the input image is grayscale; if not, convert it to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the projection by summing pixel values along the specified axis
    sum_axis = 1 if axis == 'y' else 0
   # projection = np.sum(image, axis=sum_axis) <---------------------------------------------------------------------
    cs_image, projection = cumSum(image)
    if DEBUG:
        cv2.imwrite(os.path.join(os.getcwd(), 'debug', 'segmentation', 'cs_image.jpg'), cs_image)
        logger.trace(f"Saved image with initial break points to: {os.path.join(os.getcwd(), 'debug', 'segmentation', 'cs_image.jpg')}")
        with open(os.path.join(os.getcwd(), 'debug', 'segmentation', 'cs_image.csv'), 'w') as file:
            file.write("Row, Cumulative Sum\n")
            for row_num, value in enumerate(projection):
                file.write(f"{row_num}, {value}\n")

    logger.trace(f"Projection: {projection}")

    logger.debug(f"Locating Maxima of {len(projection)} points above {round(thresholdRate, 2)}% of max value")

    # Ensure thresholdRate is within 0 to 100%
    if thresholdRate > 100:
        logger.error(f"Threshold rate {thresholdRate}% is greater than 100%. Setting threshold rate to 100%.")
        thresholdRate = 100
    elif thresholdRate < 0:
        logger.error(f"Threshold rate {thresholdRate}% is less than 0%. Setting threshold rate to 0%.")
        thresholdRate = 0

    # Ensure distance is non-negative
    if distance is not None and distance < 0:
        logger.error(f"Distance {distance} is negative. Setting distance to None.")
        distance = None

    # Ensure prominence is non-negative
    if prominence is not None and prominence < 0:
        logger.error(f"Prominence {prominence} is negative. Setting prominence to None.")
        prominence = None

    # Calculate the absolute threshold value based on the percentage of the max value
    max_value = np.max(projection)
    threshold = max_value * (thresholdRate / 100)

    logger.trace(f"Calculated threshold value: {threshold} (from {thresholdRate}% of max value {max_value})")


    # Manually find peaks
    logger.trace(f"Finding Break Points in {len(projection)} points)")
    peaks = []
    for i in range(1, len(projection) - 1):
        if projection[i] > threshold and projection[i] > projection[i - 1] and projection[i] > projection[i + 1]:
            peaks.append(i)
    logger.trace(f"Found {len(peaks)} candidate Break Points (peaks) in {len(projection)} points)")


    # Apply prominence filter if specified
    
    if prominence:
        pre_prom = len(peaks)
        op = prominence 
        prominence = max(projection) * (prominence/100) # Refactor prominence to be a percent of the max peak
        logger.info(f"prominence set to:{prominence} ({op}%), max: {max(projection)}, filtering {pre_prom} break points") 
        valid_peaks = []
        for peak in peaks:
            left_base = peak
            while left_base > 0 and projection[left_base] > projection[left_base - 1]:
                left_base -= 1
            right_base = peak
            while right_base < len(projection) - 1 and projection[right_base] > projection[right_base + 1]:
                right_base += 1
            peak_prominence = projection[peak] - max(projection[left_base], projection[right_base])
            if peak_prominence >= prominence:
                valid_peaks.append(peak)
        peaks = valid_peaks
        pf_loss = len(peaks)/pre_prom * 100
        Statistics.set_statistic(axis + "_pf_loss", pf_loss)
        logger.debug(f"prominence filter removed {pre_prom - len(peaks)} of {pre_prom} break points, leaving {len(peaks)}") 

    # Apply distance filter if specified
    if distance:
        pre_dist = len(peaks)
        logger.debug(f"Filtering {len(peaks)} by distance of {distance}")
        filtered_peaks = []
        last_peak = -distance  # Initialize to a value to allow the first peak
        for peak in peaks:
            if peak - last_peak >= distance:
                filtered_peaks.append(peak)
                last_peak = peak
        peaks = filtered_peaks
        df_loss = len(peaks)/pre_dist * 100
        Statistics.set_statistic(axis + "_df_loss", df_loss)
        logger.debug(f"distance filter removed {pre_dist - len(peaks)} of {pre_dist} break points, leaving {len(peaks)}")

    peaks = np.array(peaks)
    ibp_loss = (1 - len(peaks) / len(projection)) * 100
    Statistics.set_statistic(axis + '_ibp_loss' ,  ibp_loss) 

    if len(peaks) == 0:
        logger.warning(f"No peaks found above threshold {threshold}. Returning all indices of values above the threshold.")
        peaks = np.where(projection > threshold)[0]  # Return all indices where projection is above threshold
    elif ibp_loss > RuntimeParameters.max_ibp_loss:
        peaks = np.where(projection > threshold)[0]  # Return all indices where projection is above threshold
        logger.warning(f"Locating peaks resulted in too few break points, loss: {ibp_loss}%. Returning all {len(peaks)} candidate peaks above the threshold. Try lowering threshold rate")
    else:
        logger.trace(f"Returning {len(peaks)} of {len(projection)} points, loss: {ibp_loss:.2f}%")

    Statistics.set_statistic(axis + '_init_peaks', len(peaks))

    return peaks, projection

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
    logger.trace(f"Break Points: {data}")

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
    Statistics.chp_loss = chp_loss = round((1 - len(clusters) / len(data)) * 100, 2)
    logger.debug(f"Returning {len(clusters)} clusters; eps={eps}, min_samples={min_samples}, effective loss={chp_loss}%.")
    logger.trace(f"Clusters: {clusters}")
    
    return clusters, labels
    
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

def split_large_breaks(int_list, threshold):
    """
    Split large breaks in the list based on the given threshold.
    
    Args:
        int_list (list of int): List of breakpoints.
        threshold (int): Threshold value to determine large breaks.
    
    Returns:
        list: A sorted list of breakpoints with large breaks split.
    """
    additions = []

    # Insert a leading zero for correct initial diff calculation
    int_list.insert(0, 0)
    for i in range(len(int_list) - 1):
        # Calculate the difference between consecutive breakpoints
        diff = abs(int_list[i + 1] - int_list[i])
        if diff > threshold:
            # If the difference is larger than the threshold, calculate the midpoint
            avg = int(round((int_list[i + 1] + int_list[i]) / 2, 0))
            additions.append(avg)

    # Remove the leading zero before extending
    int_list.pop(0)
    
    # Extend the original list with the new elements
    int_list.extend(additions)

    # Sort the list
    sorted_list = sorted(int_list)
    logger.trace(f"Split large breaks with threshold {threshold}: {sorted_list}")
    return sorted_list

def find_breakpoints(clusters, data=None, method='index_average', threshold=50, split_large_breaks_flag=False):
    """
    Find breakpoints in clusters using the specified method ('y_max', 'y_average', or 'index_average').
    
    Args:
        clusters (list of list of int): List of clusters, where each cluster is a list of indices.
        data (list or np.array, optional): List of values corresponding to the data indices (histogram values or y values). Default is None.
        method (str): Method to use for finding breakpoints ('y_max', 'y_average', or 'index_average'). Default is 'index_average'.
        threshold (int): Threshold value to determine large breaks for the 'y_average' method.
        split_large_breaks_flag (bool): Flag to determine if large breaks should be split.
    
    Returns:
        list: A list of breakpoints.
    """
    breakpoints = []
    logger.info(f"Starting breakpoint finding with method '{method}'")

    if method == 'y_max':
        if data is None:
            raise ValueError("Data is required for method 'y_max'")
        # Find breakpoints based on the maximum value in each cluster
        for cluster in clusters:
            if len(cluster) == 0:
                continue
            max_value = -np.inf
            index = -1
            for i in cluster:
                if data[i] > max_value:
                    max_value = data[i]
                    index = i
            breakpoints.append(index)
        logger.debug(f"y_max method: Found {len(breakpoints)} breakpoints.")
    elif method == 'y_average':
        if data is None:
            raise ValueError("Data is required for method 'y_average'")
        # Find breakpoints based on the value closest to the average value in each cluster
        for cluster in clusters:
            y_values = [data[value] for value in cluster]
            average_y = np.mean(y_values)
            closest_value = min(cluster, key=lambda x: abs(data[x] - average_y))
            breakpoints.append(closest_value)
        logger.debug(f"y_average method: Found {len(breakpoints)} breakpoints.")
    elif method == 'index_average':
        # Find breakpoints based on the average of the actual indexes in each cluster
        for cluster in clusters:
            if len(cluster) == 0:
                continue
            average_index = int(round(np.mean(cluster)))
            breakpoints.append(average_index)
        logger.debug(f"index_average method: Found {len(breakpoints)} breakpoints.")
    else:
        raise ValueError("Invalid method. Use 'y_max', 'y_average', or 'index_average'.")

    if split_large_breaks_flag:
        original_breakpoints = breakpoints.copy()
        breakpoints = split_large_breaks(breakpoints, threshold)
        logger.info(f"Split large breaks: {original_breakpoints} -> {breakpoints}")

    logger.success(f"Completed breakpoint finding with method '{method}'")
    logger.trace(f"Final breakpoints: {breakpoints}")
    return breakpoints

def find_shortest_path_new(image, gradients, start, axis='y', log_cost_factor=15, bias_factor=10, gradient_factor=5):
    """
    Find the shortest path in a binary image using a modified Dijkstra's algorithm with exponential and bias costs.

    Arguments:
    image -- np.ndarray, the binary image (Image must be grayscale with one channel)
    gradients -- np.ndarray, the gradient magnitudes of the image
    start -- int, the starting coordinate (row for y-axis, column for x-axis)
    axis -- str, the axis along which to find the path ('y' for rows, 'x' for columns)
    log_cost_factor -- float, the factor for logarithmic cost (penalizes vertical movements based on the exponential distance)
    bias_factor -- float, the factor for bias cost (penalizes deviations from the starting row)
    gradient_factor -- float, the factor for gradient cost (penalizes crossing high-gradient regions, discouraging the path from crossing text lines or words)

    Returns:
    path -- list of tuples, the sequence of points in the shortest path
    """
    logger.debug(f"Finding Shortest Path from: {start}")
    logger.trace(f"start: {start}, axis={axis}, log_cost_factor={log_cost_factor}, bias_factor={bias_factor}, gradient_factor={gradient_factor}")
    
    if not (len(image.shape) == 2):
        logger.critical("Image provided is not grayscale or has too many channels, exiting")
        exit(100)

    rows, cols = image.shape

    # Check if the start index is within valid bounds
    if (axis == 'y' and (start < 0 or start >= rows)) or (axis == 'x' and (start < 0 or start >= cols)):
        logger.critical(f"Start index {start} is out of bounds for axis '{axis}' with size {rows if axis == 'y' else cols}")
        raise ValueError(f"Start index {start} is out of bounds for axis '{axis}' with size {rows if axis == 'y' else cols}")

    if axis == 'y':
        size1, size2 = rows, cols
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Directions: up, down, left, right
    else:
        size1, size2 = cols, rows
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Directions: left, right, up, down

    logger.trace(f"size1: {size1}, size2: {size2}, directions: {directions}")

    # Initialize distance and previous node arrays
    dist = np.full((size1, size2), np.inf, dtype=np.float64)
    prev = np.full((size1, size2), None, dtype=object)

    # Set the starting distance and initialize the priority queue
    if axis == 'y':
        dist[start, 0] = float(image[start, 0])
        queue = [(dist[start, 0], start, 0)]
    else:
        dist[0, start] = float(image[0, start])
        queue = [(dist[0, start], 0, start)]


    while queue:
        curr_dist, pos1, pos2 = heappop(queue)
        

        if pos2 == size2 - 1:
            logger.debug(f"Reached the end of the axis:{axis} at position: ({pos1}, {pos2})")
            break

        for d1, d2 in directions:
            new_pos1, new_pos2 = pos1 + d1, pos2 + d2
            if 0 <= new_pos1 < size1 and 0 <= new_pos2 < size2:
                log_cost = log_cost_factor * np.log1p(abs(new_pos1 - pos1))
                bias_cost = bias_factor * abs(new_pos1 - start)
                grad_cost = gradient_factor * gradients[new_pos1, new_pos2] if axis == 'y' else gradients[new_pos2, new_pos1]
                new_dist = curr_dist + (float(image[new_pos1, new_pos2]) if axis == 'y' else float(image[new_pos2, new_pos1])) + log_cost + bias_cost + grad_cost
                
                if new_dist < dist[new_pos1, new_pos2]:
                    dist[new_pos1, new_pos2] = new_dist
                    prev[new_pos1, new_pos2] = (pos1, pos2)
                    heappush(queue, (new_dist, new_pos1, new_pos2))
                    

    path = []
    min_dist = np.inf
    min_pos1 = -1

    for i in range(size1):
        if dist[i, size2 - 1] < min_dist:
            min_dist = dist[i, size2 - 1]
            min_pos1 = i

    if min_pos1 == -1:
        logger.warning("No valid path found")
        return path

    logger.debug(f"Minimum distance at end of axis: {min_dist}, Position: {min_pos1}")

    pos1, pos2 = min_pos1, size2 - 1

    while (pos1, pos2) != (start, 0):
        path.append((pos1, pos2) if axis == 'y' else (pos2, pos1))
        pos1, pos2 = prev[pos1, pos2]
        #logger.trace(f"Backtracking: current position ({pos1}, {pos2})")

    path.append((start, 0) if axis == 'y' else (0, start))
    path.reverse()

    logger.debug(f"Path found for start point: {start}")

    return path

def findTextSeperation(baseImage, seg_config: dict, axis='y'):
    """
    Find paths in the binary image using thresholding, clustering, and shortest path algorithms.

    Arguments:
    baseImage -- np.ndarray, the input binary image
    threshRate -- float, the threshold rate to determine break points
    eps -- int, the maximum distance between two samples for one to be considered as in the neighborhood of the other (DBSCAN parameter)
    min_samples -- int, the number of samples (or total weight) in a neighborhood for a point to be considered as a core point (DBSCAN parameter)
    axis -- str, the axis along which to find paths ('y' for rows, 'x' for columns)

    Returns:
    paths_found -- list of lists of tuples, the found paths
    """
    RuntimeParameters.update_parameters(seg_config)
    RuntimeParameters.display

    height, width = baseImage.shape[:2]
    logger.debug(f"Base image dimensions: height={height}, width={width}")

    
    peaks, projection = findInitialBreakPoints(baseImage, axis, 
                                               RuntimeParameters.threshRate, 
                                               distance=RuntimeParameters.distance,
                                               prominence=RuntimeParameters.prominence)
    
    # Debugging: create image with initial break points
    if DEBUG:
        initImg = cv2.cvtColor(baseImage.copy(), cv2.COLOR_GRAY2BGR)
        for peak in peaks:
            if axis == 'y':
                initImg = cv2.line(initImg, (0, peak), (initImg.shape[1], peak), color=(0, 0, 255), thickness=2)
                cv2.putText(initImg, f"Yinit: {peak}", (5, peak - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
            else:
                initImg = cv2.line(initImg, (peak, 0), (peak, initImg.shape[0]), color=(0, 0, 255), thickness=2)
                cv2.putText(initImg, f"Xinit: {peak}", (peak + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
        cv2.imwrite(os.path.join(os.getcwd(), 'debug', 'segmentation', 'dbgInitialBreakPoints.jpg'), initImg)
        logger.trace(f"Saved image with initial break points to: {os.path.join(os.getcwd(), 'debug', 'segmentation', 'dbgInitialBreakPoints.jpg')}")

    logger.trace(f"Finding clusters of break points. eps:{RuntimeParameters.eps}, min_samples:{RuntimeParameters.min_samples}")
    clusters, labels = group_by_proximity(peaks, RuntimeParameters.eps, RuntimeParameters.min_samples)

    if DEBUG:
        c = Colors.RED
        clustImg = cv2.cvtColor(baseImage.copy(), cv2.COLOR_GRAY2BGR)
        for cluster in clusters:
            for peak in cluster:
                if axis == 'y':
                    clustImg = cv2.line(clustImg, (0, peak), (clustImg.shape[1], peak), color=c, thickness=2)
                else:
                    clustImg = cv2.line(clustImg, (peak, 0), (peak, clustImg.shape[0]), color=c, thickness=2)
            c = Colors.BLUE if c == Colors.RED else Colors.RED
        cv2.imwrite(os.path.join(os.getcwd(), 'debug', 'segmentation', 'dbgBreakPointClusters.jpg'), clustImg)
        logger.trace(f"Saved image with break points clusters to: {os.path.join(os.getcwd(), 'debug', 'segmentation', 'dbgBreakPointClusters.jpg')}")

    peaks = find_breakpoints(clusters, projection, method='y_max')

    if DEBUG:
        clustImgPeaks = cv2.cvtColor(baseImage.copy(), cv2.COLOR_GRAY2BGR)
        c = Colors.RED
        for peak in peaks:
            if axis == 'y':
                clustImgPeaks = cv2.line(clustImgPeaks, (0, peak), (clustImgPeaks.shape[1], peak), color=c, thickness=2)
                cv2.putText(clustImgPeaks, f"Yclust: {peak}", (25, peak - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, Colors.BLUE, 2)
            else:
                clustImgPeaks = cv2.line(clustImgPeaks, (peak, 0), (peak, clustImgPeaks.shape[0]), color=c, thickness=2)
                cv2.putText(clustImgPeaks, f"Xclust: {peak}", (peak + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, Colors.BLUE, 2)
            c = Colors.BLUE if c == Colors.RED else Colors.RED
        cv2.imwrite(os.path.join(os.getcwd(), 'debug', 'segmentation', 'dbgClusterPeaks.jpg'), clustImgPeaks)
        logger.trace(f"Saved image with peak break points clusters to: {os.path.join(os.getcwd(), 'debug', 'segmentation', 'dbgClusterPeaks.jpg')}")

    gradients = calculate_gradients(baseImage)
    thresh_image = cv2.bitwise_not(baseImage)

    paths_found = []
    pathImg = baseImage.copy()

    if axis == 'y':
        valid_peaks = [peak for peak in peaks if 0 <= peak < height]
        valid_peaks.append(height - 1)
    else:
        valid_peaks = [peak for peak in peaks if 0 <= peak < width]
        valid_peaks.append(width - 1)
    logger.trace(f"Found {len(valid_peaks)} valid peaks.")

    for peak in valid_peaks:
        shortest_path = find_shortest_path_new(thresh_image, gradients, peak, axis=axis, log_cost_factor=200, bias_factor=100, gradient_factor=5)
        paths_found.append(shortest_path)

    return paths_found

def colTransitionPoints(image, threshold=3):
    """
    Find all x-values in an image that mark where black pixels start or stop, with a threshold for minimum spacing.

    Args:
        image (np.ndarray): The input binary image (black and white).
        threshold (int): Minimum number of columns without black pixels to consider a transition.

    Returns:
        List[int]: List of x-values where black pixels start or stop.
    """
    height, width = image.shape
    transitions = []
    in_black_region = False
    last_transition = -threshold  # Initialize to a value to allow the first transition

    for x in range(width):
        column = image[:, x]
        has_black_pixels = np.any(column == 0)

        if has_black_pixels and not in_black_region:
            # Transition from white to black
            in_black_region = True
            if x - last_transition >= threshold:
                transitions.append(x)
                last_transition = x
        elif not has_black_pixels and in_black_region:
            # Transition from black to white
            in_black_region = False
            if x - last_transition >= threshold:
                transitions.append(x)
                last_transition = x

    return transitions

def cumSum(image):
    """
    Create a color image where each pixel represents the cumulative sum of all previous pixels in the source image,
    distributed evenly across the three color channels. Also, create a matrix with the row number and the final sum for each row.
    
    Args:
        image (np.ndarray): The input image (grayscale or color).
        
    Returns:
        np.ndarray: The output color image with cumulative sums distributed across channels.
        np.ndarray: The matrix with row numbers and final cumulative sums.
    """
    # Convert to grayscale if the image is not already
    if len(image.shape) == 3:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        grayscale_image = image
    else:
        raise ValueError("Input image must be either a grayscale or a color image")
    
    # Invert the grayscale image
    #grayscale_image = cv2.bitwise_not(grayscale_image)
    
    # Compute the maximum sum a row could have
    height, width = grayscale_image.shape
    max_sum = width * 255
    
    # Thresholds for each channel
    threshold1 = max_sum / 3
    threshold2 = 2 * threshold1
    
    # Initialize the cumulative sum and color image
    color_image = np.zeros((*grayscale_image.shape, 3), dtype=np.uint8)
    cumulative_sums = []

    for row in range(height):
        cumulative_sum = 0
        for col in range(width):
            pixel_value = grayscale_image[row, col]
            cumulative_sum += pixel_value
            if cumulative_sum <= threshold1:
                color_image[row, col, 0] = int((cumulative_sum / threshold1) * 255)
            elif cumulative_sum <= threshold2:
                color_image[row, col, 0] = 255
                color_image[row, col, 1] = int(((cumulative_sum - threshold1) / threshold1) * 255)
            else:
                color_image[row, col, 0] = 255
                color_image[row, col, 1] = 255
                color_image[row, col, 2] = int(((cumulative_sum - threshold2) / threshold1) * 255)
        cumulative_sums.append(cumulative_sum)
    
    cumulative_sums_matrix = np.array(cumulative_sums, dtype=np.int64)
    
    return color_image, cumulative_sums_matrix