import cv2
import numpy as np
from e2cr_plotting import *
from itertools import groupby
from scipy.signal import find_peaks
import jenkspy

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKRED = '\033[31m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

HY_AXIS = 1
WX_AXIS = 0
COL_HIST = 0
ROW_HIST = 1

def find_top_n_peaks(peaks, peak_heights, n):
    """
    Find the top N peaks based on their heights.
    
    Args:
        peaks (list): Indices of the peaks.
        peak_heights (list): Heights of the peaks.
        n (int): Number of top peaks to return.
    
    Returns:
        list: Indices of the top N peaks.
    """
    sorted_indices = np.argsort(peak_heights)[-n:][::-1]
    return peaks[sorted_indices]



def find_peaks_above_threshold(data, height, distance=None, prominence=None):

    print(f"{Colors.OKBLUE}\nfFinding peeks in {len(data,)} points, min height:{height}, distance: {distance} will bw selected{Colors.ENDC}")
    """
    Find peaks in the data that are above the specified threshold.
    
    Args:
        data (list or np.array): The input data array.
        threshold (float): The threshold value.
        distance (int, optional): Minimum horizontal distance (in samples) between neighboring peaks.
        prominence (float, optional): Required prominence of peaks.
    
    Returns:
        list: Indices of peaks above the threshold.
    """
    peaks, _ = find_peaks(data, height, distance, prominence=prominence)
    return peaks






def findHistSplit(ImageToHist, histAxis,threshRate = .96,expectedTextHeight=80):
    print(f"{Colors.OKBLUE}\nfindHistSplit(splitImage, histAxis = {histAxis},threshRate = {threshRate},expectedTextHeight={expectedTextHeight}){Colors.ENDC}")
    print(f"{Colors.OKBLUE}\tReceived Image with: w:{ImageToHist.shape[1]}, h:{ImageToHist.shape[0]}{Colors.ENDC}")
    if ImageToHist is None or ImageToHist.size == 0: 
        print("Image data missing")
        return []
    if ImageToHist.shape[0] <= expectedTextHeight:
        print("\tImage is already at or under expected height/width")
        return []
    
    # Convert to grayscale
    ImageToHist = cv2.cvtColor(ImageToHist, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"e:/OCR2/lines/ImageToHist_gray.jpg", ImageToHist)
    #ImageToHist = cv2.adaptiveThreshold(ImageToHist, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 4)
    
    
    ImageToHist = cv2.fastNlMeansDenoising(ImageToHist, None, 19, 21, 21 )
    cv2.imwrite(f"e:/OCR2/lines/ImageToHist_denois.jpg", ImageToHist)

    # Calculate histograms (axis 1 horizontal, axis zero vertical)
    if histAxis in (1,0):
        image_hist = np.sum(ImageToHist, histAxis)
    else:
        raise IndexError(f'axis for split must be {ROW_HIST} (History of rows) or {COL_HIST} (Hist of Columns), axis={histAxis} was passed')
    print(f"{Colors.OKBLUE}\tHistorgram Complete: max:{max(image_hist)}, min:{min(image_hist)}, avg:{round(sum(image_hist)/len(image_hist),2)}{Colors.ENDC}")
    
    
    # thresh - threshold to consider blank space across axis
    # lower will be more splits, need to find a function to pick this based on type/sxis/hist or something else, 
    # possibly itterate at larger numbers (96% seams optimal for our data set)
    # may have to change as iterate though slices, getting closer to individual words
    
    histThreshold = max(image_hist) * threshRate
    print(f"{Colors.OKBLUE}\tThreshold set to: {histThreshold}, filtering Histogram{Colors.ENDC}")

    # filter list by hisThreshold
    lines = [y for y, value in enumerate(image_hist) if value > histThreshold]

    # calculate mid point of spaces between words/lines
    def filter_break_points(numbers, expectedTextHeight):
        print(f"{Colors.OKGREEN}\t\t Filtering {len(numbers)} break points from Histogram, with average height of {expectedTextHeight}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}\t\t Recieved{len(numbers)}{Colors.ENDC}")
        # Function to identify sequences
        
        #if not numbers:
        #return [], []  # Return empty lists if input is empty

        sequences = []
        current_sequence = []

        for i in range(len(numbers) - 1):
            current_sequence.append(numbers[i])
            if abs(numbers[i] - numbers[i + 1]) > expectedTextHeight:
                if current_sequence:
                    sequences.append(current_sequence)
                    current_sequence = []
        current_sequence.append(numbers[-1])  # Include the last number in the last sequence
        if current_sequence:
            sequences.append(current_sequence)

        averages = [sum(seq) / len(seq) for seq in sequences]
        #averages = [max(seq) for seq in sequences]

        averages = [int(x) for x in averages]
        #print("Sequences:", grouped_numbers)
        print(f"{Colors.OKGREEN}\t\t Filtering Complete: {len(averages)} break points remain of {len(numbers)}, with assumed height of {expectedTextHeight}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}\t\t New Break Points: {averages}{Colors.ENDC}")
        return sequences, averages 

    def find_natural_breaks(data, num_classes):
        """
        Find natural breaks in the data using Jenks natural breaks optimization.
        
        Args:
            data (list): List of numerical data.
            num_classes (int): Number of classes to divide the data into.
        
        Returns:
            tuple: A tuple containing a list of lists (segments) and a list of break points.
        """
        breaks = jenkspy.jenks_breaks(data, num_classes)
        segments = []
        for i in range(len(breaks) - 1):
            segment = [val for val in data if breaks[i] <= val < breaks[i + 1]]
            segments.append(segment)
        print(f"{Colors.OKGREEN}\t\t Found: {len(segments)}, and {len(breaks)}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}\t\t Segment {segments}, and Breaks:{breaks}{Colors.ENDC}")
        return segments, breaks
    
    

    #fuz_dat = [10,11,12,25,26]
    #sequences, averages = filter_break_points(fuz_dat,15)
    sequences, averages = filter_break_points(lines,expectedTextHeight) 
    
    
    averages = find_peaks_above_threshold(image_hist,histThreshold,expectedTextHeight,100)
    lst = []
    for avg in averages:
        lst.append(image_hist[avg])
    print(f"Count of Averages {len(averages)}, Len of Ldt {len(lst)}")
    averages = find_top_n_peaks(averages,lst,25)
    plot_histogram_with_peaks(image_hist,averages,histThreshold)

    
    #segments, breaks = find_natural_breaks(averages,int(expectedTextHeight/5))
    sequences, averages = filter_break_points(averages,expectedTextHeight) 
    #plot_numbers(image_hist,marked_x_values=averages,threshold_value=[histThreshold])
    
    adpThresh = draw_horizontal_lines(ImageToHist.copy(),averages)
    cv2.imwrite(f"e:/OCR2/lines/adapted.jpg", adpThresh)
    
    averages = [int(round(num, 1)) for num in averages]
    print(f"{Colors.OKBLUE}\tReturning {len(averages)} break points: {averages}{Colors.ENDC}")
    return averages





