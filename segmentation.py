import cv2
import numpy as np
from e2cr_plotting import *
from scipy.signal import find_peaks
from common import *
from sklearn.cluster import DBSCAN
from scipy import stats
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm

def getImageHist(ImageToHist, histAxis):
    #print(f"{Colors.OKBLUE}\nfindHistSplit(splitImage, histAxis = {histAxis}")
    #print(f"{Colors.OKBLUE}\tReceived Image with: w:{ImageToHist.shape[1]}, h:{ImageToHist.shape[0]}{Colors.ENDC}")
    if ImageToHist is None or ImageToHist.size == 0: 
        print(f"{Colors.WARNING}\tImage data missing")
        return []
    
    
    # Convert to grayscale if needed
    if len(ImageToHist.shape) == 3 and ImageToHist.shape[2] == 3:
        ImageToHist = cv2.cvtColor(ImageToHist, cv2.COLOR_BGR2GRAY)
    
    
    # Calculate histograms (axis 1 horizontal, axis zero vertical)
    if histAxis in (1,0):
        image_hist = np.sum(ImageToHist, histAxis)
    else:
        raise IndexError(f'axis for split must be {ROW_HIST} (History of rows) or {COL_HIST} (Hist of Columns), axis={histAxis} was passed')
        return np.array(1,1)
    #print(f"{Colors.OKBLUE}\tHistorgram Complete{Colors.ENDC}")
    
    return image_hist

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


def group_by_proximity_dbscan(data, eps, min_samples):
    if len(data) < 1: 
        print(f"{Colors.FAIL}\tE2CR Segmentation: Grouping by Proximity of {len(data)} points.{Colors.ENDC}")
        return [], np.ndarray(1)
    #print(f"{Colors.OKBLUE}\nE2CR Segmentation: Grouping by Proximity of {len(data)} points.{Colors.ENDC}")
    #print(f"{Colors.OKBLUE}\t eps={eps}, min_samples={min_samples}{Colors.ENDC}")
    """
    Group numbers in a list by their proximity to each other using DBSCAN clustering.
    
    Args:
        data (list or np.array): List of numerical data.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    
    Returns:
        tuple: A tuple containing a list of lists (clusters) and the labels for each point.
    """
    data = np.array(data).reshape(-1, 1)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    labels = clustering.labels_
    
    clusters = []
    for label in set(labels):
        if label != -1:  # -1 is the noise label
            cluster = data[labels == label].flatten().tolist()
            clusters.append(cluster)

    chp_loss = (1 - len(cluster) / len(data)) * 100
    #print(f"{Colors.OKBLUE}E2CR Segmentation: Grouping by Proximity Returning {len(clusters)} clusters of {len(data)} data points, loss={chp_loss}.{Colors.ENDC}")
    return clusters, labels


def find_breakpoints_max(clusters, imageHist):
    ##print(f"{Colors.OKBLUE}E2CR Segmentation: Finding Max in {len(clusters)} Clusters.{Colors.ENDC}")
    """
    Find the breakpoints in the clusters based on the highest value in imageHist.
    
    Args:
        clusters (list): List of clusters.
        imageHist (list or np.array): List of histogram values corresponding to the data indices.
    
    Returns:
        list: A list of breakpoints.
    """
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
        BreakValues.append(imageHist[value])
    #print(f"{Colors.OKBLUE}E2CR Segmentation: Found Max in {len(clusters)} Clusters, returning {len(BreakPoints)} break points.{Colors.ENDC}")
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

    breakpoints = split_large_breaks(breakpoints, DefaultSeg.EXPECTED_IMAGE_HW * 2)
    

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
    

def human_classify(input_folder=r'E:\E2CR\crops'):
    """
    Process images in the input folder, asking the user to input the text each image says.
    Create a folder under /training named after the user input and move the image to that folder.
    If the user inputs nothing, delete the file.

    Args:
        input_folder (str): The folder containing images to process.
    """
    training_folder = r'E:\E2CR\training'

    # Create the training folder if it doesn't exist
    if not os.path.exists(training_folder):
        os.makedirs(training_folder)

    # Iterate through each image in the input folder
    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)

        # Check if the file is an image
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue

        # Ask the user for input
        user_input = input(f"What does the image '{filename}' say? ")

        # If the user inputs nothing, delete the file
        if not user_input.strip():
            os.remove(filepath)
            print(f"Deleted '{filename}'")
        else:
            # Create the target folder if it doesn't exist
            target_folder = os.path.join(training_folder, user_input.strip())
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            # Move the image to the target folder
            shutil.move(filepath, create_unique_filepath(os.path.join(target_folder, filename)))
            print(f"Moved '{filename}' to '{target_folder}'")

        
