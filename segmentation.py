import cv2
import numpy as np
from common import *
from sklearn.cluster import DBSCAN
from loguru import logger
from heapq import heappop, heappush
import matplotlib.pyplot as plt
from tqdm import tqdm



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

def findInitialBreakPoints(image: np.ndarray, axis: str = DefaultParameters.axis,
                           useCumulativeSum: bool = DefaultParameters.useCumulativeSum,                            
                           threshRate: float = DefaultParameters.threshRate, 
                           distance: int = DefaultParameters.distance, 
                           findpeaks = DefaultParameters.findPeaks, 
                           prominence: float = DefaultParameters.prominence,
                           max_threshRate_loss: float = DefaultParameters.max_threshRate_loss,
                           usefilterValsByProximity: bool = DefaultParameters.usefilterValsByProximity) -> tuple[np.ndarray, np.ndarray]:
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
    logger.trace((f"findInitialBreakPoints(image: np.ndarray, axis: {axis}, useCumulativeSum: {useCumulativeSum},threshRate: {threshRate}, distance: {distance}, findPeaks: {findpeaks}, prominence: {prominence}: max_ibp_loss: {max_threshRate_loss}"))
   

    # Check if the input axis is valid
    if axis not in ['x', 'y']:
        logger.critical(f"Improper axis passed: {axis}")
        raise ValueError(f"Improper axis passed: {axis}")

    # Check if the input image is grayscale; if not, convert it to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the projection by summing pixel values along the specified axis
    sum_axis = 1 if axis == 'y' else 0

    if useCumulativeSum:
        logger.info(f"Projecttion is Using Cumulative Sum" )
        cs_image, projection = cumSum(image)
        if DEBUG:
            cv2.imwrite(os.path.join(DEBUG_FOLDER, 'cs_image.jpg'), cs_image)
            logger.trace(f"Saved image with initial break points to: {os.path.join(DEBUG_FOLDER, 'cs_image.jpg')}")
            with open(os.path.join(DEBUG_FOLDER, 'cs_image.csv'), 'w') as file:
                file.write("Row, Cumulative Sum\n")
                for row_num, value in enumerate(projection):
                    file.write(f"{row_num}, {value}\n")
    else:
        projection = np.sum(image, axis=sum_axis)

    peaks, _ = filterValsByThreshold(projection,threshRate=threshRate,max_theshRate_loss=max_threshRate_loss)     # Filter by threshold
    if DEBUG:
        threshold_peaks = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        for peak in peaks:
            cv2.line(threshold_peaks, (0, peak), (image.shape[1], peak), Colors.RED, 2)
        cv2.imwrite(os.path.join(DEBUG_FOLDER, 'threshold_peaks.jpg'), threshold_peaks)

    if usefilterValsByProximity:
        peaks, _ = filterValsByProximity(peaks)                                     # filter out consecutive 'peaks'
        if DEBUG:
            Proximity_peaks = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
            for peak in peaks:
                cv2.line(Proximity_peaks, (0, peak), (image.shape[1], peak), Colors.RED, 2)
            cv2.imwrite(os.path.join(DEBUG_FOLDER, 'Proximity_peaks.jpg'), Proximity_peaks)
    
    # Ensure distance is non-negative
    if distance is not None and distance < 0:
        logger.error(f"Distance {distance} is negative. Setting distance to None.")
        distance = None

    logger.critical(f"Snnity Check: len(peaks): {len(peaks)}")
  

    # Ensure prominence is non-negative
    if prominence is not None and prominence < 0:
        logger.error(f"Prominence {prominence} is negative. Setting prominence to None.")
        prominence = None

    # Manually find peaks
    if findpeaks:
        pass
        # Need to impliment something better here or disregard


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
            else:
                logger.debug(f"Removed {peak} which was {peak - last_peak} from {last_peak}")
        peaks = filtered_peaks
        df_loss = len(peaks)/pre_dist * 100
        Statistics.set_statistic(axis + "_df_loss", df_loss)
        logger.debug(f"distance filter removed {pre_dist - len(peaks)} of {pre_dist} break points, leaving {len(peaks)}")

    #peaks = np.array(peaks)
    ibp_loss = ((1 - len(peaks)) / len(projection)) * 100
    logger.debug(f"Returning {len(peaks)} of {len(projection)}, {ibp_loss} % loss")
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

    breakpoints, _ = normalizeValsByProximity(breakpoints)

    logger.success(f"Completed breakpoint finding with method '{method}'")
    logger.trace(f"Final breakpoints: {breakpoints}")
    return breakpoints

def find_shortest_path_new(image, gradients, start, axis='y', log_cost_factor=DefaultParameters.log_cost_factor, bias_factor=DefaultParameters.bias_factor, gradient_factor=DefaultParameters.gradient_actor):
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

def findTextSeperation(baseImage, axis='y',
                       threshRate: int = DefaultParameters.threshRate,
                        distance = DefaultParameters.distance,
                        prominence = DefaultParameters.prominence,
                        eps = DefaultParameters.eps,
                        min_samples = DefaultParameters.min_samples,
                        method = DefaultParameters.method,
                        log_cost_factor = DefaultParameters.log_cost_factor,
                        bias_factor = DefaultParameters.bias_factor,
                        gradient_factor = DefaultParameters.gradient_actor
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

    height, width = baseImage.shape[:2]
    logger.debug(f"Base image dimensions: height={height}, width={width}")
  
    peaks, projection = findInitialBreakPoints(baseImage,threshRate=threshRate) # US################ need to pass through arguments !
    
    # Debugging: create image with initial break points
    if DEBUG:
        if len(baseImage.shape) == 2:
            initImg = cv2.cvtColor(baseImage.copy(), cv2.COLOR_GRAY2BGR)
        elif baseImage.shape[2] == 3:
            initImg = baseImage.copy()
        else:
            raise ValueError("Unsupported number of channels in the input image: {baseImage.shape}")

        for peak in peaks:
            if axis == 'y':
                initImg = cv2.line(initImg, (0, peak), (initImg.shape[1], peak), color=(0, 0, 255), thickness=2)
                cv2.putText(initImg, f"Yinit: {peak}", (5, peak - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
            else:
                initImg = cv2.line(initImg, (peak, 0), (peak, initImg.shape[0]), color=(0, 0, 255), thickness=2)
                cv2.putText(initImg, f"Xinit: {peak}", (peak + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
        cv2.imwrite(os.path.join(DEBUG_FOLDER, 'dbgInitialBreakPoints.jpg'), initImg)
        logger.trace(f"Saved image with initial break points to: {os.path.join(DEBUG_FOLDER, 'dbgInitialBreakPoints.jpg')}")

    logger.trace(f"Finding clusters of break points. eps:{eps}, min_samples:{min_samples}")
    clusters, labels = group_by_proximity(peaks, eps, min_samples)

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
        cv2.imwrite(os.path.join(DEBUG_FOLDER, 'dbgBreakPointClusters.jpg'), clustImg)
        logger.trace(f"Saved image with break points clusters to: {os.path.join(DEBUG_FOLDER, 'dbgBreakPointClusters.jpg')}")

    peaks = find_breakpoints(clusters, projection, method=method)

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
        cv2.imwrite(os.path.join(DEBUG_FOLDER, 'dbgClusterPeaks.jpg'), clustImgPeaks)
        logger.trace(f"Saved image with peak break points clusters to: {os.path.join(DEBUG_FOLDER, 'dbgClusterPeaks.jpg')}")

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
        shortest_path = find_shortest_path_new(thresh_image, gradients, peak, axis=axis, log_cost_factor=log_cost_factor, bias_factor=bias_factor, gradient_factor=gradient_factor)
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

def cumSum(image, subOnDecrease: bool = DefaultParameters.subOnDecrease):
    """
    Create a color image where each pixel represents the cumulative sum of all previous pixels in the source image,
    distributed evenly across the three color channels. Also, create a matrix with the row number and the final sum for each row.
    
    Args:
        image (np.ndarray): The input image (grayscale or color).
        
    Returns:
        np.ndarray: The output color image with cumulative sums distributed across channels.
        np.ndarray: The matrix with row numbers and final cumulative sums.
    """
    logger.trace(f"cumSum(image, subOnDecrease: {subOnDecrease})")
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

    pbar = tqdm(total=height*width, desc="Building Cumulative Sum of Image")
    for row in range(height):
        cumulative_sum = 0
        
        for col in range(width):
            pixel_value = grayscale_image[row, col]
            if col > 0:
                if grayscale_image[row, col] < grayscale_image[row, col-1] and subOnDecrease:
                    cumulative_sum -= pixel_value
                else:
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
            pbar.update(1)
        cumulative_sums.append(cumulative_sum)
        
    
    cumulative_sums_matrix = np.array(cumulative_sums, dtype=np.int64)
    if DEBUG: cv2.imwrite(os.path.join(DEBUG_FOLDER, 'cumsum.jpg'), color_image)

    
    return color_image, cumulative_sums_matrix

def filterValsByProximity(values, proximity: int = DefaultParameters.f_proximity):
    logger.trace(f"Filtering values in a list {len(values)} values long, by a proximity of {proximity}")

    if not len(values) > 1:
        return [], 0.0

    result = []
    start = values[0]
    last_v = values[0]

    for v in values[1:]:
        if v > last_v + proximity:
            # End of current group
            result.extend([start, last_v])
            # Start new group
            start = v
        last_v = v
    
    # Append the last group
    result.extend([start, last_v])

    # Calculate loss percentage
    unique_values = np.unique(values)
    filtered_values = np.unique(result)
    loss_percentage = (1 - len(filtered_values) / len(unique_values)) * 100

    logger.trace(f"Filtering values returning {len(result)} values. {loss_percentage:.2f}% loss")
    return result, loss_percentage

def filterValsByThreshold(values, threshRate: float = DefaultParameters.threshRate, max_theshRate_loss: float = DefaultParameters.max_threshRate_loss):
    logger.trace(f"filterValsByThreshold(values, threshRate: {threshRate})")

    # Ensure thresholdRate is within 0 to 100%
    if 0 >= threshRate > 100:
        logger.error(f"Threshold rate {threshRate}% is not valid, setting to defualt: {DefaultParameters.threshRate}.")
        threshRate = DefaultParameters.threshRate

    # Calculate the absolute threshold value based on the percentage of the max value
    max_value = np.max(values)
    threshold = max_value * (threshRate / 100)

    logger.trace(f"Calculated threshold value: {threshold} (from {threshRate}% of max value {max_value})")
    new_v = [i for i in range(len(values)) if values[i] > threshold]
    # Calculate the number and percentage of filtered values
    filtered_count = len(values) - len(new_v)
    filtered_percentage = (filtered_count / len(values)) * 100

    logger.trace(f"Threshold filtered: {filtered_count} from projection ({len(values)}, {filtered_percentage:.2f}% loss)")


    if filtered_percentage > max_theshRate_loss:
        logger.warning(f"Locating peaks resulted in too few break points, loss: {filtered_percentage}%. Returning all {len(values)} origional values. Try lowering threshold rate")
        new_v = values
    else:
        logger.trace(f"Returning {len(new_v)} of {len(values)} points, loss: {filtered_percentage:.2f}%")
    return new_v, filtered_percentage

def normalizeValsByProximity(values, proximity: int = DefaultParameters.min_white_space):
    logger.trace(f"Normalizing values in a list {len(values)} values long, by a proximity of {proximity}")

    if not values:
        return [], 100.0

    start_len = len(values)
    groups = []
    start = values[0]
    last_v = values[0]

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
            normalized_values.append(int((group[0] + group[1]) / 2))
        else:
            normalized_values.extend([group[0], group[1]])

    # Calculate loss percentage
    unique_values = np.unique(normalized_values)
    loss_percentage = (1 - len(unique_values) / start_len) * 100

    logger.trace(f"Normalizing values returning {len(unique_values)} values. {loss_percentage:.2f}% loss")
    return unique_values, loss_percentage

