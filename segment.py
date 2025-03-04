import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Iterator
from test5 import find_indices
import math

class SegmentPreProcSettings:
    def __init__(self):
        self.adaptiveThreshold = True
        self.calcAdaptiveParamaters = True
        self.adapBlockSize =  11
        self.adapC = 21

    def calculateAdapParameters(self,image):
        self.adapBlockSize, self.adapC =  self._calcBlockC(image)
    
    def _calcBlockC(image):
        mean_intensity = np.mean(image)
        std_deviation = np.std(image)
        median_intensity = np.median(image)
        std_val = np.std(image)
        block_size = int(2 * (std_deviation // 2) + 1)  # Must be an odd number
        c_value = abs(mean_intensity - median_intensity)
        if c_value < 15: c_value = 15
        return block_size,  c_value
        

class SegmentationStatistics:
    def __init__(self):
        self.filterIndexByValThresholdLoss = np.inf

    def show(self):
        print("\033[94m---------------------------------------------")
        print("|          Segmentation Statistics          |")
        print("---------------------------------------------")
        print(f"\033[92m| filterIndexByValThresholdLoss      {self.filterIndexByValThresholdLoss:.2f}% |")
        
class Segment:
    def __init__(self, image: np.array=None, label="New Segment",x=0,y=0,segtype='Document',parent=None):
        if image is None:
            raise ValueError("No image provided to the segment")
        self.Statistics = SegmentationStatistics()

        self.segStartIndexes = []
        self.textRowHeight = 15
        # Initialize Default Attributes
        self.isSubSegment = parent
        if not segtype in ['Document','Row','CC','Contour']:
            raise ValueError("Uknown Type")
        self.type = segtype
        self.x = x
        self.width = 0
        self.y = y
        self.height = 0
        self.is_segmented = False
        self.is_pre_processed = False
        self._segments = []
        self._contours = []
        self.label = label
        self._is_color = False
        self._col_hist  = []
        self._row_hist = []
        self._gray_image: np.array = image
        if image is not None:
            self.image = image.copy()
            self._is_color = len(image.shape) == 3
            if self._is_color:               
                self._gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            self.height, self.width = self._gray_image.shape
            self._calcPixelPercentages()

            # Calculate the Histogram for Grayscale Copy of Image
            if self.ink == 'white':     
                self._row_hist = np.sum(np.bitwise_not(self._gray_image), axis=1)
                self._col_hist = np.sum(np.bitwise_not(self._gray_image), axis=0)
            else:
                self._row_hist = np.sum(self._gray_image, axis=1)
                self._col_hist = np.sum(self._gray_image, axis=0)

            self._calcPixelPercentages()
            #self._getTextBoundries()
        if parent:
            self.parent = parent
        else:
            self.parent = None

    def preProcess(self):
        if not self._gray_image is None:
            pass       

    def Segment(self, lines=None,axis='y'):
        self.segpath = lines.copy()
        
        if self.image is None or self.image.size == 0:
            raise ValueError("Input image is empty or invalid")
        
        subSegments = []
        subSegmentCount = 0
        if lines:
            for i in range(len(lines)):
                mask_pts = []
                if i == 0:
                    if axis == 'y':
                        #logger.debug(f"Processing from top of the image to line {i}")
                        y_start = 0
                        _, y_end = self._getLineBoundries(lines[i])
                        mask_pts.extend([[0, 0]])
                        mask_pts.extend([[x, y] for y, x in lines[i]])
                        mask_pts.extend([[self.width, 0]])
                    else:
                        #logger.debug(f"Processing from left of the image to line {i}")
                        x_start = 0
                        x_end, _ = self._getLineBoundries(lines[i])
                        mask_pts.extend([[0, 0]])
                        mask_pts.extend([[x, y] for y, x in lines[i]])
                        mask_pts.extend([[0, self.height]])
                elif i == len(lines) - 1:
                    if axis == 'y':
                        #logger.debug(f"Processing between line {i-1} and bottom of the image")
                        _, y_start = self._getLineBoundries(lines[i-1])
                        y_end = self.height
                        mask_pts.extend([[x, y] for y, x in lines[i-1]])
                        mask_pts.extend([[x, y] for y, x in lines[i]][::-1])
                        mask_pts.extend([[self.width, self.height], [0, self.height]])
                    else:
                        #logger.debug(f"Processing between line {i-1} and right of the image")
                        x_start, _ = self._getLineBoundries(lines[i-1])
                        x_end = self.width
                        mask_pts.extend([[x, y] for y, x in lines[i-1]])
                        mask_pts.extend([[x, y] for y, x in lines[i]][::-1])
                        mask_pts.extend([[self.width, 0], [self.width, self.height]])
                else:
                    if axis == 'y':
                    #logger.debug(f"Processing between line {i-1} and line {i}")
                        y_start, _ = self._getLineBoundries(lines[i-1])
                        _, y_end = self._getLineBoundries(lines[i])
                        mask_pts.extend([[x, y] for y, x in lines[i-1]])
                        mask_pts.extend([[x, y] for y, x in lines[i]][::-1])
                    else:
                        #logger.debug(f"Processing between line {i-1} and line {i}")
                        x_start, _ = self._getLineBoundries(lines[i-1])
                        x_end, _ = self._getLineBoundries(lines[i])
                        mask_pts.extend([[x, y] for y, x in lines[i-1]])
                        mask_pts.extend([[x, y] for y, x in lines[i]][::-1])

                # Create a mask for the area between the current line and the next line
                mask = np.zeros_like(self._gray_image, dtype=np.uint8)
                pts = np.array(mask_pts, dtype=np.int32)
                o_mask = cv2.fillPoly(mask, [pts], 255)
                
                # Apply the mask to the original image to extract the area between the lines
                masked_image = cv2.bitwise_and(o_mask, self.image, mask=mask)
                masked_image = cv2.bitwise_not(masked_image)
                masked_image = cv2.bitwise_and(masked_image, o_mask, mask=o_mask)
                masked_image = cv2.bitwise_not(masked_image)
                
                # Crop the bounding box of the masked region
                if axis == 'y':
                    subSegment = masked_image[y_start:y_end, 0:self.width]
                else:
                    subSegment = masked_image[0:self.height, x_start:x_end]

                # Ensure the cropped image is not empty
                if subSegment.size > 0:
                    subSegments.append(Segment(image=subSegment,label=self.label + f"_{subSegmentCount:03d}_",segtype='Row',parent=True,y=y_start))
                    subSegmentCount += 1
            if len(subSegments) > 1: 
                self._segments = subSegments
                self.is_segmented = True
            else:
                self._segments = []
        return self._segments.copy()
            
    def _getLineBoundries(self,line):
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
        min_x = min(x for y, x in line)
        max_x = max(x for y, x in line) 

        return min_y, max_y

    def _calcPixelPercentages(self):
        white_pixel_count = np.sum(self._gray_image == 255)
        black_pixel_count = np.sum(self._gray_image == 0)   
        
        total_pixel_count = self.height * self.width

        # Calculate the percentage of white pixels
        self.percentWhite = (white_pixel_count / total_pixel_count) 
        self.percentBlack = (black_pixel_count / total_pixel_count)

        self.ink = 'black' if black_pixel_count < white_pixel_count else 'white'

    def show(self):
        cv2.imshow(self.label,self.image)
        cv2.waitKey()

    def showHist(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15), constrained_layout=True)
        fig.suptitle(self.label)
        # Plot the image on the first subplot
        ax1.imshow(self.image, aspect='equal', cmap='gray')
                   

        # Scale the histogram to match the image dimensions
        hist_scaled = self._col_hist / (np.max(self._col_hist) / self.height)

        # Create a secondary y-axis for the histogram
        ax1_hist = ax1.twinx()

        # Overlay the original histogram on the secondary y-axis
        ax1_hist.plot(self._col_hist, color='blue')  # Adjust color as needed

        # Set the x and y limits to match the image dimensions
        ax1.set_xlim(0, self.image.shape[1])
        ax1.set_ylim(0, self.image.shape[0])
        ax1_hist.set_ylim(0, np.max(self._col_hist))

        # Invert the y-axis of the image plot to match the image orientation
        ax1.invert_yaxis()

        # Set the title and labels
        ax1.set_title('Column Sums (Swapped Axes)')
        ax1.set_xlabel('Column Index')
        ax1.set_ylabel('Image Pixel Value')
        ax1_hist.set_ylabel('Original Histogram Value')

        # Plot the image on the second subplot
        ax2.imshow(self.image, aspect='equal',cmap='gray')

        # Set the y limits to match the image dimensions
        ax2.set_ylim(0, self.height)

        # Set the x limits to match the histogram dimensions
        ax2.set_xlim(0, self.width)

        # Invert the y-axis to match the image orientation
        ax2.invert_yaxis()

        # Create a secondary x-axis at the top for the pixel index
        ax2_top = ax2.twiny()

        # Set the x limits of the top x-axis to match the image dimensions
        ax2_top.set_xlim(0, np.max(self._row_hist))
        # Flip and plot the row histogram
        r = np.flip(self._row_hist)
        ax2_top.plot(r, np.arange(len(self._row_hist))[::-1], color='blue')
        ax2.axhline(y=np.mean(self._row_hist), color='red', linestyle='--', label='avg first wp from top')

        # Set the title and labels
        ax2.set_title('Row Histogram')
        ax2.set_ylabel('Row Index')
        ax2.set_xlabel('Sum of Row Pixel Values')
        ax2_top.set_xlabel('Pixel Index')

        plt.show()

    def ShowStats(self):
            self.Statistics.show()

    def __str__(self):
        if not self.parent:
            parent = 'none'
        else:
            parent = self.parent.label
        return f"Segment:'{self.label}' >> {self.ink} ink, x:{self.x},y:{self.y}, width:{self.width}, height: {self.height}, parent: {parent}, {len(self._segments)} child segments, {100*self.percentWhite:.2f}% white pixels."

    def __iter__(self) -> Iterator[Segment]:
        self._index = 0  # Reset the index when creating a new iterator
        return self

    def __next__(self) -> Segment:
        if self._index < len(self._segments):
            subSegment = self._segments[self._index]
            self._index += 1
            return subSegment
        
        else:
            raise StopIteration
        
    def _getTextBoundries(self):
                
        # Initialize lists to store y-values
        fromTop = [int(self.height/2)] * self.width
        fromBottom = [int(self.height/2)] * self.width
        searchPixel = 255 if self.ink == 'white' else 0

        # Iterate through columns
        for x in range(self.width):
            # Find the first white pixel from the top
            for y in range(self.height):
                if self._gray_image[y, x] == searchPixel:  # Assuming white pixel is represented by 255
                    fromTop[x] = y
                    break
            
            # Find the first white pixel from the bottom
            for y in range(self.height - 1, -1, -1):
                if self._gray_image[y, x] == searchPixel:  # Assuming white pixel is represented by 255
                    fromBottom[x] = y
                    break
            
            self.topBoundry = fromTop
            self.bottomBoundry = fromBottom
        
        t, b = find_indices(self._row_hist, np.mean(self._row_hist)*.5)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15), constrained_layout=True)
        fig.suptitle("text Boundries")
        # Plot the image on the first subplot
        ax1.imshow(self.image, aspect='equal', cmap='gray')
        ax1.plot(self.topBoundry, color='blue', linestyle='--', label='Boundry from top')
        ax1.axhline(y=np.mean(self.topBoundry), color='green', linestyle='-', label='avg first wp from top')
        ax1.axhline(y=t, color='yellow', linestyle='--', label='avg first wp from top')
        ax1.axhline(y=np.mean(self.bottomBoundry), color='green', linestyle='-', label='avg first wp from top')
        ax1.axhline(y=b, color='yellow', linestyle='--', label='avg first wp from top')
        
        
        ax1.set_xlim(0, self.image.shape[1])
        ax1.set_ylim(0, self.image.shape[0])
        ax1.invert_yaxis()
        ax1.set_title('Column Sums (Swapped Axes)')
        ax1.set_xlabel('Column Index')
        ax1.set_ylabel('Image Pixel Value')

        ax2.set_ylim(0, self.height)
        ax2.set_xlim(0, max(self._row_hist))
        ax2.invert_yaxis()
        ax2_top = ax2.twiny()
        ax2_top.set_xlim(0, np.max(self._row_hist))
        r = np.flip(self._row_hist)
        ax2.plot(r, np.arange(len(self._row_hist))[::-1], color='blue')
        ax2.axvline(x=np.mean(self._row_hist), color='red', linestyle='--', label='Mean Sum(Rows Pixel Values)')
        
        ax2.set_title('Row Histogram')
        ax2.set_ylabel('Row Index')
        ax2.set_xlabel('Sum of Row Pixel Values')

        ax2.axhline(y=t, color='red', linestyle='--', label='avg first wp from top')
        ax2.axhline(y=b, color='red', linestyle='--', label='avg first wp from top')
        
        
        return fromTop, fromBottom
    
    def calcTextCore(self, threshold):
        
        arr = arr.astype(np.int64)
        n = len(arr)
        max_index = np.argmax(arr)  # Find the index of the maximum value
        max_value = np.max(arr)
        

        left_index = max_index
        right_index = max_index

        nm = 0
        ni = max_index
        for m in range(len(arr)-1, -1, -1):
            if arr[m] > max_value * .70:
                if arr[m] < nm:
                    break
                nm = arr[m]
                ni =  m
        
        max_index = ni
        max_value = arr[m]


        delta = int(max_value*.05)

        for i in range(max_index, -1, -1):
            if arr[i] - arr[left_index]  > delta or arr[i] < threshold:
                break
            left_index = i
            

        # Iterate forwards from the max_index to find the first index where the value falls below the threshold
        for j in range(max_index, n):

            if arr[j]-arr[right_index] > delta or arr[j] < threshold:

                break
            right_index = j

        return left_index, right_index
    
    def deskew(self):
        """
        Deskews the input image by detecting and correcting the skew angle.

        Parameters:
        img (np.ndarray): Input grayscale image to be deskewed.

        Returns:
        np.ndarray: Deskewed image.
        """
        
        if self.ink == 'black':
            img = np.bitwise_not(self._gray_image.copy())    
        else:
            img = self._gray_image.copy()
            
        
        # Add a black border around the image
        img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        # Detect edges using the Canny edge detector
        edges = cv2.Canny(img, 50, 200, apertureSize=3)
        
        # Define a kernel for morphological operations
        kernel = np.ones((3, 3), np.uint8)
        
        # Apply dilation and erosion to the image
        dilated = cv2.dilate(edges, kernel, iterations=1)
        dilated = cv2.erode(dilated, kernel, iterations=1)

        # Detect lines using the Hough Line Transform
        lines = cv2.HoughLines(dilated, 10, np.pi/1000, 10)
        tlines = []
        line_image = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        angles = []

        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                delta_x = x2 - x1
                delta_y = y2 - y1
                
                # Calculate angle in radians
                angle_rad = math.atan2(delta_y, delta_x)
                
                # Convert angle to degrees
                angle_deg = math.degrees(angle_rad)
                
                # Filter for angles within a specific range
                if -43 < angle_deg < -40:
                    print(f"Detected line angle: {angle_deg}")
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    tlines.append(line)
                    angles.append(angle_deg)
        

        # If no valid angles are found, return the original image
        if len(angles) == 0:
            print(1)
            return self._gray_image  
        
        # Calculate the average angle for deskewing
        angle_avg = abs(sum(angles) / len(angles))

        angle_rad = np.deg2rad(angle_avg)
        
        # Get the image dimensions
        (h, w) = img.shape[:2]
        
        # Calculate the skew transformation matrix
        skew_matrix = np.array([
            [1, np.tan(angle_rad), 0],
            [0, 1, 0]
        ], dtype=np.float32)
        
        # Apply the affine transformation to deskew the image
        deskewed_image = cv2.warpAffine(img, skew_matrix, (w + 50, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        if self.ink == 'black':
            print(2)
            self._gray_image = np.bitwise_not(deskewed_image)
            return self._gray_image
        else:
            print(3)
            self._gray_image = deskewed_image
            return self._gray_image
    
    def __lt__(self, other):
        if self.type in ['cc','word','contour']:
            return self.x < other.x
        elif self.type == 'row':
            return self.y < other.y
    
    def extract_cc(self):
        # Load the image
        image = self._gray_image
        
        if self.ink == 'black':
            image = cv2.bitwise_not(image)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
        
        print(f"Number of connected components found: {num_labels - 1}")  # Subtract 1 to exclude the background component


        # Extract and display each connected component
        connected_components = []
        for label in range(1, num_labels):  # Start from 1 to exclude the background
            # Create a mask for the current component
            component_mask = np.zeros_like(image)
            component_mask[labels == label] = 255

            # Find the bounding box coordinates
            x, y, w, h, area = stats[label]

            # Extract the component from the original image using the mask
            component_image = cv2.bitwise_and(image, image, mask=component_mask)
            connected_component  = Segment(component_image[y:y+h, x:x+w],segtype='CC',x=x,y=y,parent=self)
            connected_components.append(connected_component)
            connected_components = sorted(connected_components)
            for i, cc in enumerate(connected_components):
                cc.label = self.label + f" Connected Component {i}"

            self.connected_compnents = connected_components
        return connected_components
            
    

    '''
        Segmentation Methods
    '''
    def _filterIndexByValThreshold(self, valThresholdPercent):
        if  0 >= valThresholdPercent > 1:
            ...
        
        if self.type ==  'Document':
            max_value = np.max(self._row_hist)
            values = self._row_hist
        else:
            max_value = np.max(self._col_hist)
            values = self._col_hist

        threshold_value = max_value * valThresholdPercent
        self.threshold_value = threshold_value
        
        
        return_values = [[i, v] for i, v in enumerate(values) if v > threshold_value]
        print(f"Filtered {len(values)-len(return_values)} out of {len(values)} values, having higher than {threshold_value}, {valThresholdPercent}% of {max_value}")

        self.indexesFilteredByValThreshold = True
        self.filterIndexByValThresholdResults = return_values
        print(f"before Setting Starts: {self.segStartIndexes=}")
        if len(self.segStartIndexes) == 0:
            print(f"Setting Starts: {self.segStartIndexes=}")
            self.segStartIndexes = return_values # If not Candidate Start Indexes exist for Segmentation, make these start points

        self.Statistics.filterIndexByValThresholdLoss = ((len(values) - len(return_values)) / len(values)) * 100
         
        return return_values
    
    def _filterMinNConsecutive(self, maxProximity):
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
        print(f"self.segStartIndexes: {self.segStartIndexes}")
        values = [i for i, v in self.segStartIndexes]
        
        return_values = []
        start = 0  # Start index of the current sequence

        # Iterate through the list starting from the second element
        for i in range(1, len(values)):
            # Check if the current value is not consecutive to the previous one
            if values[i] != values[i - 1] + 1:
                # Check if the current sequence length meets the minimum consecutive count
                if i - start >= maxProximity:
                    # Add the start and stop values of the sequence to the result
                    return_values.append([values[start], values[i - 1]])
                # Update the start index to the current index
                start = i

        # Check the last sequence after the loop ends
        if len(values) - start >= maxProximity:
            # Add the start and stop values of the last sequence to the result
            return_values.append([values[start], values[-1]])

        # Calculate the loss percentage
        
        self.Statistics.filterMinNConsecutiveLoss = ((len(values) - len(return_values)) / len(values)) * 100

        self.filterMinNConsecutiveResults = return_values
        return return_values
    
    def _calcAverage_distance(self):
        if len(self.filterMinNConsecutiveResults) < 2:
            return 0, []  # Return 0 and an empty list if there are fewer than 2 pairs
        
        distances = []
        for i in range(len(self.filterMinNConsecutiveResults) - 1):
            distance = abs(self.filterMinNConsecutiveResults[i][1] - self.filterMinNConsecutiveResults[i + 1][0])
            distances.append(distance)
    
        average_distance = sum(distances) / len(distances)
        print("\n",distances)
        return average_distance, distances

    def _calculateTextRowHeight(self):
        consec_area_bounds = self._filterMinNConsecutive(3)
        avg, dist = self._calcAverage_distance()
        distance = int(avg + np.std(dist))
        
        self.textRowHeight = distance
        return distance
        

    '''
        Display Values
    '''
    def showProperties(self):
        print("\033[94m---------------------------------------------")
        print("|                 Properties                |")
        print("---------------------------------------------")
        print(f"\033[92m| textRowHeight                          {self.textRowHeight} |")
        

    def fibvtShow(self):
        if self.indexesFilteredByValThreshold:
            # Create a plot
            plt.figure(figsize=(10, 10))
            plt.imshow(self._gray_image)
            plt.axis('off')  # Hide axis

            # Plot horizontal lines at specified positions
            for i, v in self.filterIndexByValThresholdResults:
                plt.axhline(y=i, color='green', linestyle='--', linewidth=1)

            plt.title("Candidate Indexes after _filterIndexByValThreshold", fontsize=12)

            # Display the plot
            plt.show()

        if self.segStartIndexes:
            # Create a plot
            plt.figure(figsize=(10, 10))
            plt.imshow(self._gray_image)
            plt.axis('off')  # Hide axis

            # Plot horizontal lines at specified positions
            for i, v in self.segStartIndexes:
                plt.axhline(y=i, color='green', linestyle='--', linewidth=1)

            plt.title("Segmentation Start Indexes", fontsize=12)

            # Display the plot
            plt.show()




