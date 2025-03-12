import cv2
import numpy as np
import os
from common import Colors, DEBUG, DEBUG_FOLDER, Parameters, Statistics
from sklearn.cluster import DBSCAN
from loguru import logger
from heapq import heappop, heappush
from imageman import visual_debug, percentWhite, deskew
import csv
from imageman import draw_path_on_image, cropSegmentFromImage, cropTextFromRow,crop_image_to_content, percentWhite
from plotting import *
from concurrent.futures import ThreadPoolExecutor, as_completed
import heapq


def findLineBetweenRowsOfText2(image, start, gradients, n, axis='y',
                            log_cost_factor=Parameters.pathFinder.log_cost_factor, 
                            bias_factor=Parameters.pathFinder.bias_factor, 
                            gradient_factor=Parameters.pathFinder.gradient_factor):
    """
    Optimized function to find the shortest path in a binary image using a modified Dijkstra's algorithm 
    with exponential and bias costs, while restricting vertical movement.

    Arguments:
    image -- np.ndarray, the binary image (grayscale, one channel)
    start -- int, starting coordinate (row for y-axis, column for x-axis)
    gradients -- np.ndarray, the gradient magnitudes of the image (precomputed)
    n -- int, maximum allowed deviation (vertical movement restriction)
    axis -- str, axis along which to find the path ('y' for rows, 'x' for columns)

    Returns:
    path -- list of tuples, the sequence of points in the shortest path
    """

    logger.debug(f"Start: {start}, Vertical restriction: ±{n}")
    if len(image.shape) != 2:
        raise ValueError("Input image must be grayscale.")

    rows, cols = image.shape

    if (axis == 'y' and not (0 <= start < rows)) or (axis == 'x' and not (0 <= start < cols)):
        raise ValueError(f"Start index {start} is out of bounds.")

    size1, size2 = (rows, cols) if axis == 'y' else (cols, rows)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Precompute log values for speed
    log_cost_table = log_cost_factor * np.log1p(np.arange(size1))

    # Initialize cost and path tracking
    dist = np.full((size1, size2), np.inf, dtype=np.float64)
    prev = np.full((size1, size2, 2), -1, dtype=np.int32)  # Stores (row, col) instead of objects

    # Set the starting point
    dist[start, 0] = float(image[start, 0])
    queue = [(dist[start, 0], start, 0)]

    heapq.heapify(queue)

    while queue:
        curr_dist, pos1, pos2 = heapq.heappop(queue)

        if pos2 == size2 - 1:  # Goal reached
            break

        for d1, d2 in directions:
            new_pos1, new_pos2 = pos1 + d1, pos2 + d2

            # Restrict movement within `n` range from start
            if not (start - n <= new_pos1 <= start + n):
                continue

            if 0 <= new_pos1 < size1 and 0 <= new_pos2 < size2:
                log_cost = log_cost_table[abs(new_pos1 - pos1)]
                bias_cost = bias_factor * abs(new_pos1 - start)
                grad_cost = gradient_factor * gradients[new_pos1, new_pos2] if axis == 'y' else gradients[new_pos2, new_pos1]

                new_dist = curr_dist + float(image[new_pos1, new_pos2]) + log_cost + bias_cost + grad_cost

                # Speed up queue processing
                if new_dist < dist[new_pos1, new_pos2] * 0.95:  
                    dist[new_pos1, new_pos2] = new_dist
                    prev[new_pos1, new_pos2] = [pos1, pos2]
                    heapq.heappush(queue, (new_dist, new_pos1, new_pos2))

    # Backtrack to construct the path
    path = []
    min_pos1 = np.argmin(dist[:, size2 - 1])

    if dist[min_pos1, size2 - 1] == np.inf:
        logger.warning("No valid path found.")
        return path

    pos1, pos2 = min_pos1, size2 - 1
    while prev[pos1, pos2][0] != -1:  # Stop when backtracking reaches start
        path.append((pos1, pos2) if axis == 'y' else (pos2, pos1))
        pos1, pos2 = prev[pos1, pos2]

    path.append((start, 0) if axis == 'y' else (0, start))
    path.reverse()

    logger.debug(f"Start={start}, Path Length={len(path)}, End={min_pos1}, Cost={dist[min_pos1, size2 - 1]}")
    return path

def average_consecutive_distance(values):
    if len(values) < 2:
        return 0  # No distance if there's only one or zero elements
    
    distances = [abs(values[i+1] - values[i]) for i in range(len(values) - 1)]
    return sum(distances) / len(distances)

def findPathStartPoints(image: np.ndarray, axis: str = Parameters.axis,
                        expectedTextRowHeight: int = Parameters.expectedTextRowHeight,
                           useCumulativeSum: bool = Parameters.useCumulativeSum,                            
                           threshRate: float = Parameters.Segmentation.threshRate, 
                           distance: int = Parameters.distance, 
                           prominence: float = Parameters.prominence,
                           max_threshRate_loss: float = Parameters.max_threshRate_loss,
                           usefilterValsByProximity: bool = Parameters.usefilterValsByProximity) -> tuple[np.ndarray, np.ndarray]:
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
    
    fname = Statistics.fileName

    # Check if the input axis is valid
    if axis not in ['x', 'y']:
        logger.critical(f"Improper axis passed: {axis}")
        raise ValueError(f"Improper axis passed: {axis}")

    # Check if the input image is grayscale; if not, convert it to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the projection by summing pixel values along the specified axis
    sum_axis = 1 if axis == 'y' else 0
    projection = np.sum(image, axis=sum_axis)

    #calculate_projection_profile_and_peaks(image)

    peaks, Statistics.filterIndexByValThreshold_loss = filterIndexByValThreshold(projection,threshRate=threshRate,max_theshRate_loss=max_threshRate_loss)     # Filter by threshold
    if DEBUG:
        threshold_peaks = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        for peak, value in peaks:
            cv2.line(threshold_peaks, (0, peak), (image.shape[1], peak), Colors.RED, 2)
            cv2.putText(threshold_peaks, f"y:{peak}", (25, peak - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.BLUE, 2)
        cv2.imwrite(os.path.join(DEBUG_FOLDER, f'{fname}_threshold_peaks.jpg'), threshold_peaks)

    # calculate the expected hight of text rows using results from filterIndexByValThreshold
    consec_area_bounds, loss = filterMinNConsecutive([a for a, b in peaks],3)
    Statistics.set_statistic("filterMinNConsecutive_loss",loss)
    avg, dist = average_distance(consec_area_bounds)
    distance = int(avg + np.std(dist))
    Statistics.CalculatedRowHight = distance
    Parameters.expectedTextRowHeight = distance
    logger.info(f"Parameters.expectedTextRowHeight has been set to {distance}")
    

    if usefilterValsByProximity:
        peaks, Statistics.filterValsByProximity_loss = selectIndexInGroupProximityByVal(peaks,Parameters.f_proximity,selector='v_max')                                     # filter out consecutive 'peaks'
        if DEBUG:
            Proximity_peaks = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
            for peak in peaks:
                cv2.line(Proximity_peaks, (0, int(peak)), (image.shape[1], int(peak)), Colors.RED, 2)
                cv2.putText(Proximity_peaks,f"{int(peak)}",(2,int(peak)),cv2.FONT_HERSHEY_SIMPLEX, 0.4, Colors.BLUE, 2)
            cv2.imwrite(os.path.join(DEBUG_FOLDER, f'{fname}_selectValsByProximity.jpg'), Proximity_peaks)
      
    # Apply distance filter if specified
    if distance:
        peaks, Statistics.filterByDistance_loss = filterByDistance(peaks,image,distance,expectedTextRowHeight )
        if DEBUG: visual_debug(image,axis='y',action='draw_lines',values=peaks,operation_name="filterByDistance",alternate=True)
            
    #clusters, labels = group_by_proximity(peaks, eps, min_samples)
    
    #if DEBUG: visual_debug(image,axis='y',action='draw_lines',values=peaks,operation_name="group_by_proximity",alternate=True)
    
    #peaks = findClusterBreakpoints(clusters, projection, method=find_peaks_in_cluster_method)
    #if DEBUG: visual_debug(image,axis='y',action='draw_lines',values=peaks,operation_name="find_breakpoints")
        
    #peaks = np.array(peaks)
    fpsp_loss = (1-(len(peaks) / len(projection))) * 100
    Statistics.findPathStartPoints_loss = fpsp_loss
    logger.debug(f"Returning {len(peaks)} of {len(projection)}, {fpsp_loss} % loss")
    return np.unique(peaks), projection

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
    
    logger.debug(f"Finding clusters in {len(data)} break points. eps:{eps}, min_samples:{min_samples}")
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

def findClusterBreakpoints(clusters, data=None, method='index_average', threshold=50, split_large_breaks_flag=False):
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

    breakpoints, _ = normalizeValsByProximity(breakpoints)

    logger.success(f"Completed breakpoint finding with method '{method}'")
    logger.trace(f"Final breakpoints: {breakpoints}")
    return breakpoints

def findLineBetweenRowsOfText(image, start, axis='y',
                            log_cost_factor=Parameters.pathFinder.log_cost_factor, 
                            bias_factor=Parameters.pathFinder.bias_factor, 
                            gradient_factor=Parameters.pathFinder.gradient_factor):
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
    csv_file -- str, the CSV file to write the paths data to

    Returns:
    path -- list of tuples, the sequence of points in the shortest path
    """
 
    logger.trace(f"start: {start}, axis={axis}, log_cost_factor={log_cost_factor}, bias_factor={bias_factor}, gradient_factor={gradient_factor}")
    
    if not (len(image.shape) == 2):
        logger.critical("Image provided is not grayscale or has too many channels, exiting")
        exit(100)

    rows, cols = image.shape

    gradients = calculate_gradients(image)

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

    #logger.trace(f"size1: {size1}, size2: {size2}, directions: {directions}")

    
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
            #logger.trace(f"Reached the end of the axis:{axis} at position: ({pos1}, {pos2})")
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

    pos1, pos2 = min_pos1, size2 - 1

    while (pos1, pos2) != (start, 0):
        path.append((pos1, pos2) if axis == 'y' else (pos2, pos1))
        pos1, pos2 = prev[pos1, pos2]
        #logger.trace(f"Backtracking: current position ({pos1}, {pos2})")

    path.append((start, 0) if axis == 'y' else (0, start))
    path.reverse()

    # Collect path data (start, min, max, end)
    start_pos = path[0]
    end_pos = path[-1]
    min_pos = min(path, key=lambda x: x[1] if axis == 'y' else x[0])
    max_pos = max(path, key=lambda x: x[1] if axis == 'y' else x[0])
    total_length = len(path)
    sum_distances = sum(np.linalg.norm(np.array(path[i]) - np.array(path[i-1])) for i in range(1, len(path)))
    avg_gradient = np.mean([gradients[p[0], p[1]] for p in path] if axis == 'y' else [gradients[p[1], p[0]] for p in path])

    if len(path) < cols:
        logger.warning(f"The path found is shorter than axis, path length:{len(path)}, axis size:{cols}")

    # Write path data to CSV
    if DEBUG:
        with open(os.path.join(DEBUG_FOLDER, 'path_data.csv'), mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([start_pos, min_pos, max_pos, end_pos, total_length, sum_distances, avg_gradient])

    logger.debug(f"Starting at {start}, distance of axis is: {len(path)}, value of: {min_dist}, ending at: {min_pos1}")
    return path

def findTextRowSeperationPaths(image, axis='y',
                       threshRate: int = Parameters.Segmentation.threshRate,
                        log_cost_factor = Parameters.pathFinder.log_cost_factor,
                        bias_factor = Parameters.pathFinder.bias_factor,
                        gradient_factor = Parameters.pathFinder.gradient_factor
                          ): 
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
    

    height, width = image.shape[:2]
    logger.debug(f"Base image dimensions: height={height}, width={width}")
  
    peaks, projection = findPathStartPoints(image,threshRate=threshRate) # US################ need to pass through arguments !
    
    
    # = cv2.bitwise_not(baseImage)

    paths_found = []
    pathImg = image.copy()

    if axis == 'y':
        valid_peaks = [peak for peak in peaks if 0 <= peak < height]
        valid_peaks.append(height - 1)
    else:
        valid_peaks = [peak for peak in peaks if 0 <= peak < width]
        valid_peaks.append(width - 1)
    logger.trace(f"Found {len(valid_peaks)} valid peaks.")
    

    for peak in valid_peaks:
        shortest_path = findLineBetweenRowsOfText(image, peak, axis=axis, log_cost_factor=log_cost_factor, bias_factor=bias_factor, gradient_factor=gradient_factor)
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

def selectIndexInGroupProximityByVal(indexes_with_sum, proximity: int = Parameters.f_proximity, selector='v_max'):
    """
    Filters values in a list by grouping values that are within a certain proximity of each other.

    Parameters:
    indexes_with_sum (list of tuples): List of tuples where each tuple contains an index and its corresponding sum.
    proximity (int): The maximum distance between values to consider them as part of the same group.
    selector (str): Selector indicating how to handle grouping.

    Returns:
    tuple: (filtered_values, loss_percentage)
        filtered_values (list of int): The filtered list of values, including start and end of each group.
        loss_percentage (float): The percentage of values lost due to filtering.
    """
    logger.trace(f"Filtering values in a list {len(indexes_with_sum)} values long, by a proximity of {proximity}, selector: {selector}")

    # If the list has 1 or fewer elements, return an empty list and 0.0 loss percentage
    if len(indexes_with_sum) <= 1:
        return [], 0.0

    filtered_values = []  # List to store the filtered values
    start = indexes_with_sum[0][0]  # Start of the current group
    start_i = 0
    last_i = 1
    last_v = indexes_with_sum[0][0]  # Last value in the current group
    max_value = indexes_with_sum[0][1]  # Max value in the current group

    # Iterate through the values starting from the second element
    for i, i_s in enumerate(indexes_with_sum[1:]):
        logger.trace(f"Index: {i}= row:{i_s[0]}, sum:{i_s[1]}, start:{start} last_v:{last_v}")
        if i_s[0] > last_v + proximity:
            logger.trace(f"Found a group from {start} to {last_v}")
            # If the current value is farther than the proximity, end the current group
            if selector == 'v_max':
                if last_v-start>1:
                    # Append all elements that have the max value within the current group
                    max_value = max(iws[1] for iws in indexes_with_sum[start_i:last_i])
                    logger.trace(f"Finding indexes in the group with max value {max_value}")
                    #for iws in indexes_with_sum[start_i:last_i]:
                    #    print(iws)
                    max_elements = [iws[0] for iws in indexes_with_sum[start_i:last_i] if iws[1] == max_value]
                    if len(max_elements) > 1:
                        logger.trace(f"Multipe indexes in the group with max value {max_value}, using average: {np.sum(max_elements)/len(max_elements)}")
                        filtered_values.append(int(np.sum(max_elements)/len(max_elements)))
                    else:
                        filtered_values.append(indexes_with_sum[i][0])

                    logger.trace(f"Found {len(max_elements)} indexes in the group with max value {max_value}")
                else:
                    logger.trace(f"Single index: {max_value}")
                    filtered_values.append(last_v)
            else:
                # Default behavior (current logic)
                filtered_values.extend([start, last_v])  # Add start and last_v as they are

            # Start a new group
            start = i_s[0]   
            start_i = i 

        last_v = i_s[0]
        last_i = i

    # Append the last group
    if selector == 'y_max':
        max_elements = [iws[0] for iws in indexes_with_sum[start:last_v+1] if iws[1] == max_value]
        filtered_values.extend(max_elements)
    else:
        filtered_values.extend([start, last_v])  # Add start and last_v as they are

    # Calculate the unique values before and after filtering
    unique_values = np.unique([v for i, v in indexes_with_sum])
    filtered_values = np.unique(filtered_values)

    # Calculate the percentage of values lost due to filtering
    loss_percentage = (1 - len(filtered_values) / len(unique_values)) * 100
    

    logger.debug(f"Filtering values returning {len(filtered_values)} values. {loss_percentage:.2f}% loss")
    logger.trace(f"Filtered Values: {filtered_values}")

    return filtered_values, loss_percentage

def filterIndexByValThreshold(values, threshRate: float = Parameters.Segmentation.threshRate, max_theshRate_loss: float = Parameters.max_threshRate_loss):
    logger.info(f"Filtering values by Threshold of {threshRate}%")
    logger.debug(f"threshRate: {threshRate} max_theshRate_loss: {max_theshRate_loss}")

    # Ensure thresholdRate is within 0 to 100%
    if 0 >= threshRate > 100:
        logger.error(f"Threshold rate {threshRate}% is not valid, setting to defualt: {Parameters.Segmentation.threshRate}.")
        threshRate = Parameters.Segmentation.threshRate

    # Calculate the absolute threshold value based on the percentage of the max value
    max_value = np.max(values)
    threshold = max_value * (threshRate / 100)

    logger.debug(f"Calculated threshold value: {threshold} (from {threshRate}% of max value {max_value})")
    index_list = [i for i, v in enumerate(values) if v > threshold]

    # Calculate the number and percentage of filtered values
    filtered_count = len(values) - len(index_list)
    filtered_loss = (filtered_count / len(values)) * 100
    

    logger.trace(f"Threshold filtered: {filtered_count} from projection ({len(values)}, {filtered_loss:.2f}% loss)")

    if filtered_loss > max_theshRate_loss:
        logger.warning(f"Locating peaks resulted in too few break points, loss: {filtered_loss}%. Returning all {len(values)} origional values. Try lowering threshold rate")
        index_list = list(range(len(values)))
    else:
        logger.debug(f"Returning {len(index_list)} of {len(values)} points, loss: {filtered_loss:.2f}%")
        logger.trace(f"Values {index_list}")

    indexes_with_sum = []
    for i in index_list:
        indexes_with_sum.append([i, values[i]])

    return indexes_with_sum, filtered_loss

def normalizeValsByProximity(values, proximity: int = Parameters.min_white_space):
    """
    Normalizes a list of values by grouping close values and averaging them.

    Parameters:
    - values: List of integer values to be normalized.
    - proximity: An integer specifying the maximum distance between values to be considered in the same group.

    Returns:
    - normalized_values: A list of normalized values.
    - loss_percentage: The percentage of original values retained after normalization.
    """
    logger.trace(f"Normalizing values in a list {len(values)} values long, by a proximity of {proximity}")

    # If the list of values is empty, return an empty list and 100% loss
    if not values:
        return [], 100.0

    start_len = len(values)
    groups = []
    start = values[0]
    last_v = values[0]

    # Group values based on proximity
    for v in values[1:]:
        if v > last_v + proximity:
            # End of current group
            groups.append((start, last_v))
            # Start new group
            start = v
        last_v = v
    
    # Append the last group
    groups.append((start, last_v))

    normalized_values = []
    for group in groups:
        if group[1] - group[0] <= proximity:
            # If the group is within proximity, average the values
            normalized_values.append(int((group[0] + group[1]) / 2))
        else:
            # If the group is not within proximity, add both start and end values
            normalized_values.extend([group[0], group[1]])

    # Calculate loss percentage
    unique_values = np.unique(normalized_values)
    loss_percentage = (1 - len(unique_values) / start_len) * 100

    logger.trace(f"Normalizing values returning {len(unique_values)} values. {loss_percentage:.2f}% loss")
    return unique_values, loss_percentage

def filterByDistance(values, image: np.ndarray, distance, expectedTextRowHeight, mode='both'):
    """
    Filters a list of values based on their distances and an expected text row height.
    
    Parameters:
    - values: List of integer values representing detected features (e.g., peaks or break points).
    - image: A numpy array representing the image.
    - distance: An integer specifying the minimum distance between consecutive values.
    - expectedTextRowHeight: An integer specifying the expected height of text rows.
    - mode: A string specifying the filtering mode. Can be 'tooclose', 'toofar', or 'both'.
    
    Returns:
    - values: The filtered list of values.
    - df_loss: The percentage of original values retained after filtering.
    """
    # Initial count of values
    pre_dist = len(values)
    
    logger.trace(f"Initial values: {values}")
    filtered_peaks = []
    
    # Filter for values that are too close to each other
    if mode == 'tooclose' or mode == 'both':
        logger.debug(f"Filtering {len(values)} values for consecutive values closer than {distance}")
        last_peak = -distance  # Initialize to allow the first peak
        for value in values:
            if value - last_peak >= expectedTextRowHeight / 2:
                filtered_peaks.append(int(value))
                last_peak = value
            else:
                logger.debug(f"Removed {value} which was {value - last_peak} away from {last_peak}")
                           
        values = filtered_peaks
        
    # Filter for values that are too far apart
    if mode == 'toofar' or mode == 'both':
        toofar = []
        last_peak = values[0]  # Initialize to allow the first peak
        for i, value in enumerate(values):
            if value - last_peak >= expectedTextRowHeight * 2:
                toofar.append((last_peak, value))
            last_peak = value
        
        fill = []
        if len(toofar) > 0:
            logger.info(f"Found {len(toofar)} location(s) that are too far apart")
        for a, b in toofar:
            # Calculate the percentage of white pixels in the region
            pw = percentWhite(image[a:b, :])
            logger.trace(f"{a}-{b}: Percent white: {pw}")
            diff = int((b - a))
            for i in range(a + expectedTextRowHeight, b, expectedTextRowHeight):
                logger.debug(f"Adding start point at {i}, between {a} and {b} (diff: {diff}, expectedTextRowHeight*2: {expectedTextRowHeight*2})")
                fill.append(i)
        
        # Extend values with the newly calculated fill points and sort them
        values.extend(fill)
        values = sorted(np.unique(values))
    
    # Calculate the percentage of original values retained
    df_loss = (len(values) / pre_dist) * 100
    
    logger.debug(f"Distance filter removed {pre_dist - len(values)} of {pre_dist} break points, leaving {len(values)}")
    logger.trace(f"Returning filtered peaks: {values}, retention: {df_loss:.2f}%")
    
    return values, df_loss

def filterByProminence(values,prominence: int = Parameters.prominence):
    pre_prom = len(peaks)
    op = prominence 
    prominence = max(values) * (prominence/100) # Refactor prominence to be a percent of the max peak
    logger.info(f"Filter by Prominence, prominence set to:{prominence} ({op}%), max: {max(values)}, filtering {pre_prom} break points") 
    valid_peaks = []
    for peak in peaks:
        left_base = peak
        while left_base > 0 and values[left_base] > values[left_base - 1]:
            left_base -= 1
        right_base = peak
        while right_base < len(values) - 1 and values[right_base] > values[right_base + 1]:
            right_base += 1
        peak_prominence = values[peak] - max(values[left_base], values[right_base])
        if peak_prominence >= prominence:
            valid_peaks.append(peak)
    peaks = valid_peaks
    pf_loss = len(peaks)/pre_prom * 100
    
    logger.debug(f"prominence filter removed {pre_prom - len(peaks)} of {pre_prom} break points, leaving {len(peaks)}") 
    return peaks, pf_loss

def filterMinNConsecutive(values, consecutive_count):
    """
    Filters sequences of consecutive numbers from a list and returns pairs of 
    start and stop values for sequences that meet or exceed a minimum consecutive count.

    Parameters:
    values (list of int): The list of integers to filter.
    consecutive_count (int): The minimum length of consecutive sequences to be included in the result.

    Returns:
    tuple: A tuple containing the filtered list of start and stop values for each qualifying sequence,
           and the percentage of values lost during the filtering process.
    """
    
    logger.trace(f"{values =}, {consecutive_count =}")
    result = []
    start = 0  # Start index of the current sequence

    # Iterate through the list starting from the second element
    for i in range(1, len(values)):
        # Check if the current value is not consecutive to the previous one
        if values[i] != values[i - 1] + 1:
            # Check if the current sequence length meets the minimum consecutive count
            if i - start >= consecutive_count:
                # Add the start and stop values of the sequence to the result
                result.append([values[start], values[i - 1]])
            # Update the start index to the current index
            start = i

    # Check the last sequence after the loop ends
    if len(values) - start >= consecutive_count:
        # Add the start and stop values of the last sequence to the result
        result.append([values[start], values[-1]])

    # Calculate the loss percentage
    initial_length = len(values)
    filtered_length = len(result) // 2  # Each pair represents one sequence
    loss_percentage = ((initial_length - filtered_length) / initial_length) * 100

    logger.debug(f"{len(values) =}, {len(result) =}, {loss_percentage =:.2f}%")
    logger.trace(result)
    return result, loss_percentage

def average_distance(pairs):
    logger.warning(pairs)
    if len(pairs) < 2:
        return 0, []  # Return 0 and an empty list if there are fewer than 2 pairs
    
    distances = []
    for i in range(len(pairs) - 1):
        distance = abs(pairs[i][1] - pairs[i + 1][0])
        distances.append(distance)
    
    average_distance = sum(distances) / len(distances)
    return average_distance, distances

def segment_image(sourceImage: np.array,axis='y',
                       threshRate: int = Parameters.Segmentation.threshRate,
                        log_cost_factor = Parameters.pathFinder.log_cost_factor,
                        bias_factor = Parameters.pathFinder.bias_factor,
                        gradient_factor = Parameters.pathFinder.gradient_factor
                          ):
    pass
    # read / check filepast
    # preProcessFile

    logger.trace(f"segment_image: {sourceImage.size}, threshRate: {threshRate}, log_cost_factor: {log_cost_factor}, bias_factor: {bias_factor},  gradient_factor: {gradient_factor}") 
    

    start_points, projection = findPathStartPoints(sourceImage,threshRate=threshRate)

    avg_height = int(average_consecutive_distance(start_points)/2)
    

    logger.info("Starting gradient calc")
    gradients = calculate_gradients(sourceImage)
    logger.info("Done Starting gradient calc")

    paths_found  = []
    #for start_point in start_points:
    #    shortest_path = findLineBetweenRowsOfText(sourceImage, start_point, axis=axis, log_cost_factor=log_cost_factor, bias_factor=bias_factor, gradient_factor=gradient_factor)
    #    paths_found.append(shortest_path)

    def findLineBetweenRowsOfTextWrapper(args):
        source_image, start_point = args
        return findLineBetweenRowsOfText2(source_image,start_point,gradients,avg_height)
     # Use ThreadPoolExecutor for threading
    with ThreadPoolExecutor(max_workers=4) as executor:
        
        futures = []
        for start_point in start_points:
            args = (sourceImage, start_point)
            futures.append(executor.submit(findLineBetweenRowsOfTextWrapper, args))

        for future in as_completed(futures):
            try:
                result = future.result()
                paths_found.append(result)
            except Exception as e:
                print(f'Error processing start point: {e}')
    
    paths_found = sorted(paths_found, key=lambda point: point[0])
    
    Statistics.PathsFound = len(paths_found)
    
    pathsOnImage = sourceImage.copy()
    
    for path in paths_found:
        pathsOnImage = draw_path_on_image(pathsOnImage,path)
        cv2.imwrite(os.path.join(DEBUG_FOLDER, 'pathsOnImage.jpg'), pathsOnImage)
        cv2.putText(pathsOnImage,f"{path[0][0]}",(5,path[0][0]),cv2.FONT_HERSHEY_SIMPLEX, 0.4, Colors.BLUE, 2)
        


    segmented_image = []
    row_segmentations = []
    for num, RowOfText in enumerate(cropSegmentFromImage(sourceImage,paths_found)):
        segmented_row = []
        pct_white = percentWhite(RowOfText)
        
        if not pct_white == 100:
            RowOfText = deskew(255-RowOfText)
            RowOfText = 255-RowOfText

        word_breaks = colTransitionPoints(RowOfText)
        line_split_file = cv2.cvtColor(RowOfText.copy(), cv2.COLOR_GRAY2BGR)
        for n, x in enumerate(word_breaks):
            height, width, _ = line_split_file.shape
            # Draw a vertical line from (x_value, 0) to (x_value, height)
            #line_split_file = cv2.line(line_split_file, (x, 0), (x, height), (0, 255, 0), 2)
            #cv2.putText(line_split_file, f"{n:03d}", (x+1, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, Colors.BLUE, 1)
            #cv2.imwrite(os.path.join(os.getcwd(), './segmentation', f'line_{num:03d}_splits.tiff'), line_split_file )

        words = cropTextFromRow(RowOfText,word_breaks)
        for w, word in enumerate(words):
            segmented_row.append(word)
        
        row_segmentations.append(line_split_file)
        segmented_image.append(segmented_row)

    #visual_debug(sourceImage,values=start_points,action='draw_lines',operation_name="test")
    return segmented_image, row_segmentations

   

