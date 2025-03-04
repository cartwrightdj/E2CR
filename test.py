import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from imageman import percentWhite, draw_path_on_image, crop_image_to_content
from segmentation import normalizeValsByProximity, calculate_gradients
from common import logger, Statistics
from heapq import heappop, heappush
from collections import OrderedDict
import math

# Define the image path
image_path = r'C:\Users\User\Documents\PythonProjects\E2CR\segmentation\word_4159363_00361.jpg_032_006.tiff'




def findLineBetweenRowsOfText(image, start, axis='x',
                              log_cost_factor=1.0, 
                              bias_factor=1.0, 
                              gradient_factor=1.0,
                              deviation_penalty_factor=10.0):
    """
    Find the shortest path in a binary image using a modified Dijkstra's algorithm with exponential and bias costs.

    Arguments:
    image -- np.ndarray, the binary image (Image must be grayscale with one channel)
    start -- int, the starting coordinate (row for y-axis, column for x-axis)
    axis -- str, the axis along which to find the path ('y' for rows, 'x' for columns)
    log_cost_factor -- float, the factor for logarithmic cost (penalizes vertical movements based on the exponential distance)
    bias_factor -- float, the factor for bias cost (penalizes deviations from the starting row)
    gradient_factor -- float, the factor for gradient cost (penalizes crossing high-gradient regions, discouraging the path from crossing text lines or words)
    deviation_penalty_factor -- float, the factor for penalizing deviation from the start position

    Returns:
    path -- list of tuples, the sequence of points in the shortest path
    """

    if not (len(image.shape) == 2):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rows, cols = image.shape
    gradients = calculate_gradients(image)

    # Check if the start index is within valid bounds
    if (axis == 'y' and (start < 0 or start >= rows)) or (axis == 'x' and (start < 0 or start >= cols)):
        raise ValueError(f"Start index {start} is out of bounds for axis '{axis}' with size {rows if axis == 'y' else cols}")

    if axis == 'y':
        size1, size2 = rows, cols
        directions = [(1, 0), (-1, 0), (0, -1), (0, 1)]  # Directions: down, up, left, right
        start_pos1, start_pos2 = start, cols - 1
    else:
        size1, size2 = cols, rows
        directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # Directions: right, left, up, down
        start_pos1, start_pos2 = start, rows - 1  # Adjusted correctly for 'x' axis

    # Initialize distance and previous node arrays
    dist = np.full((size1, size2), np.inf, dtype=np.float64)
    prev = np.full((size1, size2), None, dtype=object)

    # Set the starting distance and initialize the priority queue
    if axis == 'y':
        dist[start_pos1, start_pos2] = float(image[start_pos1, start_pos2])
        queue = [(dist[start_pos1, start_pos2], start_pos1, start_pos2)]
    else:
        dist[start_pos1, start_pos2] = float(image[start_pos2, start_pos1])  # Correctly handle for 'x' axis
        queue = [(dist[start_pos1, start_pos2], start_pos1, start_pos2)]

    while queue:
        curr_dist, pos1, pos2 = heappop(queue)

        if pos2 == 0:
            break

        for d1, d2 in directions:
            new_pos1, new_pos2 = pos1 + d1, pos2 + d2
            if 0 <= new_pos1 < size1 and 0 <= new_pos2 < size2:
                log_cost = log_cost_factor * np.log1p(abs(new_pos1 - pos1))
                bias_cost = bias_factor * abs(new_pos1 - start_pos1)
                grad_cost = gradient_factor * gradients[new_pos1, new_pos2] if axis == 'y' else gradients[new_pos2, new_pos1]
                deviation_penalty = deviation_penalty_factor * abs(new_pos2 - start_pos2)
                new_dist = curr_dist + (float(image[new_pos1, new_pos2]) if axis == 'y' else float(image[new_pos2, new_pos1])) + log_cost + bias_cost + grad_cost + deviation_penalty
                
                if new_dist < dist[new_pos1, new_pos2]:
                    dist[new_pos1, new_pos2] = new_dist
                    prev[new_pos1, new_pos2] = (pos1, pos2)
                    heappush(queue, (new_dist, new_pos1, new_pos2))

    path = []
    min_dist = np.inf
    min_pos1 = -1

    for i in range(size1):
        if dist[i, 0] < min_dist:
            min_dist = dist[i, 0]
            min_pos1 = i

    if min_pos1 == -1:
        return path

    pos1, pos2 = min_pos1, 0

    while (pos1, pos2) != (start_pos1, start_pos2):
        path.append((pos1, pos2) if axis == 'y' else (pos2, pos1))
        if prev[pos1, pos2] is None:
            break
        pos1, pos2 = prev[pos1, pos2]

    path.append((start_pos1, start_pos2) if axis == 'y' else (start_pos2, start_pos1))
    path.reverse()

    return path

def filterIndexByValThreshold(values, threshhold):
    logger.info(f"Filtering values by Threshold of {threshhold = }%")
   

    index_list = [i for i, v in enumerate(values) if v < threshhold]
 
    indexes_with_sum = []
    for i in index_list:
        indexes_with_sum.append([i, values[i]])
    
    return index_list

def find_indices(arr, threshold):
    """
    Finds the indices in the array where the value first falls below the threshold when iterating
    from the index of the maximum value to the start, and then when iterating from the index of
    the maximum value to the end. If a value is close to the threshold but subsequent values move
    away from it, it uses the closest index to the threshold.

    Parameters:
    arr (np.ndarray): The input array.
    threshold (float): The threshold value to compare against.

    Returns:
    tuple: A tuple containing the index closest to the threshold when iterating backwards and the 
           first index where the value falls below the threshold when iterating forwards.
    """
    
    n = len(arr)
    max_index = np.argmax(arr)  # Find the index of the maximum value
    print(f"{max_index=} , arr[max_index]={arr[max_index]}")

    closest_exceed_index = -1
    closest_exceed_distance = float('inf')
    below_index = -1

    # Iterate backwards from the max_index to find the closest index where the value is near the threshold
    for i in range(max_index, -1, -1):
        distance = abs(arr[i] - threshold)
        if distance < closest_exceed_distance:
            closest_exceed_distance = distance
            closest_exceed_index = i
        # Break if the distance starts increasing after decreasing
        if arr[i] < threshold and (i == 0 or arr[i-1] >= threshold):
            break

    # Iterate forwards from the max_index to find the first index where the value falls below the threshold
    for j in range(max_index, n):
        if arr[j] < threshold:
            below_index = j
            break

    return closest_exceed_index, below_index

def segment_by_pct_white(image):

    
    height, width = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ia = percentWhite(image)
    print(f"Image Average: {percentWhite(image)}")
    start_x, stop_x = 0, height

    while stop_x <= width:
        print(f"{start_x = }, {stop_x = }, {percentWhite(image[0:height,start_x:stop_x])}")
        if percentWhite(image[0:height,start_x:stop_x]) > ia:
            image = cv2.line(image, (start_x, 0), (start_x, height), color=(255,0,0), thickness=2)
            image = cv2.line(image, (0, stop_x), (height, stop_x), color=(255,0,0), thickness=2)
            start_x = stop_x
            stop_x = stop_x + height
        else:
            stop_x += 1
    #cv2.imshow("seg",image)
    #cv2.waitKey()

def find_first_white_pixels(image):
    height, width = image.shape
    
    # Initialize lists to store y-values
    top_to_bottom_y_values = [-1] * width
    bottom_to_top_y_values = [-1] * width

    # Iterate through columns
    for x in range(width):
        # Find the first white pixel from the top
        for y in range(height):
            if image[y, x] == 255:  # Assuming white pixel is represented by 255
                top_to_bottom_y_values[x] = y
                break
        
        # Find the first white pixel from the bottom
        for y in range(height - 1, -1, -1):
            if image[y, x] == 255:  # Assuming white pixel is represented by 255
                bottom_to_top_y_values[x] = y
                break
    
    return top_to_bottom_y_values, bottom_to_top_y_values


def count_white_cells_below_lines(image):
    """
    Counts the number of white cells below lines drawn from each pixel along the bottom of the image to each pixel along the top of the image.
    Builds a list of tuples with (x_bottom, x_top, white_count).

    Arguments:
    image -- np.ndarray, the binary image (Image must be grayscale with one channel)

    Returns:
    counts -- list of tuples, each containing (x_bottom, x_top, white_count)
    """

    if not (len(image.shape) == 2):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rows, cols = image.shape
    results = []

    for bottom_col in range(cols):
        for top_col in range(cols):
            count = 0
            for row in range(rows - 1, -1, -1):
                col = int(bottom_col + (top_col - bottom_col) * (row / rows))
                if col < cols and image[row, col] == 255:  # Check if the cell is white
                    count += 1
            if count == 0:
                results.append((bottom_col, top_col, count))
            cv2.imshow("x",image)
            cv2.waitKey()

    return results

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Load the image in grayscale
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print(f"image is {image}")
    exit(-1)

safe = image.copy()
'''
ds_image = deskew(255-image)
image = ds_image


fig, (axOrig, axDS) = plt.subplots(2, 1, figsize=(10, 15), constrained_layout=True)

axOrig.imshow(image, cmap='gray', aspect='auto')

axDS.imshow(ds_image, cmap='gray', aspect='auto')
axDS.set_title("Deskewed Image")
plt.show()
'''



# Apply shading function
#shadedImage = shadeByDistanceBetweenInk(image, mode='cols')

# Get the width of the image
height, width = image.shape

# Calculate the sum of pixel values along each column
column_sums = np.sum(image, axis=1)
mean_x = np.mean(column_sums)
std_x = np.std(column_sums)
max_x = np.max(column_sums)
print(f"mean: {mean_x}, std:{std_x}, {mean_x+(.5*std_x)}")

a, b = find_indices(column_sums,np.mean(column_sums)-(.5*std_x))
print(f"{a= }, {b=}")

img = image[a:b,0:width]
cv2.imshow("img",img)
cv2.waitKey()
#segment_by_pct_white(img)

top_to_bottom_y_values, bottom_to_top_y_values = find_first_white_pixels(img)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), constrained_layout=True)
ax1.imshow(img, cmap='gray', aspect='auto')
##ax1.plot(top_to_bottom_y_values)
#ax1.plot(bottom_to_top_y_values)

#ax1.axhline(y=b, color='red', linestyle='-', label='from top')
#ax1.axhline(y=a, color='blue', linestyle='-', label='from bottom')

# Plot the column sums, swapping the axes and reversing the column index
r = np.flip(column_sums)
ax2.plot(r, np.arange(len(column_sums))[::-1], color='black')
ax2.set_title('Column Sums (Swapped Axes)')
ax2.set_ylabel('Column Index')
ax2.set_xlabel('Sum of Pixel Values')

ax3.imshow(image, cmap='gray', aspect='auto')
plt.show()

i_column_sums = np.sum(img, axis=0)

# Create subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), constrained_layout=True)

# Plot the original image
ax1.imshow(image, cmap='gray', aspect='auto')
ax1.set_title('Image')
ax1.axis('off')
ax1.axhline(y=b, color='red', linestyle='-', label='from top')
ax1.axhline(y=a, color='blue', linestyle='-', label='from bottom')

# Plot the column sums, swapping the axes and reversing the column index
r = np.flip(column_sums)
ax2.plot(r, np.arange(len(column_sums))[::-1], color='black')
ax2.set_title('Column Histogram')
ax2.set_ylabel('Column Index')
ax2.set_xlabel('Sum of Pixel Values')
ax2.set_ylim([0, height])
ax2.set_xlim([0, max(column_sums)])
ax2.invert_yaxis()

ax2.axhline(y=a, color='red', linestyle='-', label='from top')
ax2.axhline(y=b, color='blue', linestyle='-', label='from bottom')
ax2.axvline(x=mean_x, color='blue', linestyle='-', label='Mean x')

# Plot the standard deviation lines
ax2.axvline(x=mean_x - (.5*std_x), color='red', linestyle='--', label='Mean x - .5 STD')
ax2.legend()

ax3.imshow(image, cmap='gray', aspect='auto')
ax3.plot(i_column_sums)
ax2.set_title('Column Sums')
ax3.set_xlabel('Column Index')
ax3.set_ylabel('Sum of Pixel Values')
height, width = img.shape
ax3.set_xlim([0, width])
ax3.set_ylim([0, max(i_column_sums)])
ax3.axhline(y=int(mean_x), color='blue', linestyle='-', label='mean')
ax3.axhline(y=int(mean_x-(.5*std_x)), color='green', linestyle='--', label='-.5 std')
ax3.axhline(y=int(mean_x-std_x), color='blue', linestyle='--', label='-1 std')
plt.show()

mean_x = np.mean(i_column_sums)
std_x = np.std(i_column_sums)
max_x = np.max(i_column_sums)
print(f"{mean_x = }, {std_x =}, {max_x =}")


breaks = filterIndexByValThreshold(i_column_sums,max_x * .20)
#breaks, _ = normalizeValsByProximity(breaks,10)

#breaks,loss = filterMinNConsecutive(breaks,1)
#Statistics.set_statistic("filterMinNConsecutive_loss",{"Testing": loss})

img_a = cv2.cvtColor(safe.copy(), cv2.COLOR_GRAY2BGR)
height, width, _ = img_a.shape
new_width = width * 5
new_height = height * 5

# Resize the image
#img_a = cv2.resize(img_a, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
height, width, _ = img_a.shape
paths = []
#breaks = [[60,width-50]]
#breaks = breaks * 5
if len(breaks) == 0: print("Breaks is empty")
for brek in breaks:
    
    cv2.putText(img_a , f"Thick: ", (5,brek), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,200), 1)  
    img_a = cv2.line(img_a, (brek, 0), (brek, height), color=(0,255,255), thickness=1)
    #path = findLineBetweenRowsOfText(img_a,brek)
    #img_a = draw_path_on_image(img_a,path,thickness=1)
    #paths.append(path)
    
    
    #print("\npath",path)


print(f"{height=}, {width=}")
cv2.imshow("seg",img_a)
cv2.waitKey()

topdoen, bottomup = find_first_white_pixels(image)
tdavg = np.mean(topdoen)

# Create subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), constrained_layout=True)

# Plot the original image
ax1.imshow(image, cmap='gray', aspect='auto')
ax1.set_title('Image')
ax1.axis('off')
ax1.plot(topdoen)
ax1.plot(bottomup)
ax1.axhline(y=tdavg, color='red', linestyle='-', label='from top')
ax1.axhline(y=a, color='blue', linestyle='-', label='from bottom')
