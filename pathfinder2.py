import cv2
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import hilbert, find_peaks, savgol_filter, detrend
from scipy.ndimage import gaussian_filter1d
import heapq
from scipy.ndimage import convolve


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


def make_first_n_columns_white(image, n):
    """
    Changes the first `n` columns of an image to white.

    Arguments:
    - image: np.ndarray, input image (grayscale or color).
    - n: int, number of columns to turn white.

    Returns:
    - modified_image: np.ndarray, modified image with first `n` columns set to white.
    """
    # Ensure `n` is within valid range
    n = min(n, image.shape[1])

    # Create a copy to avoid modifying the original image
    modified_image = image.copy()

    if len(image.shape) == 2:  
        # Grayscale image
        modified_image[:, :n] = 255  # Set first `n` columns to white
    else:
        # Color image (3 channels)
        modified_image[:, :n] = (255, 255, 255)  # Set first `n` columns to white

    return modified_image

# Load grayscale image
image_path = r'C:\Users\User\Documents\PythonProjects\E2CR\debug\cany_extracted.tiff'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = make_first_n_columns_white(image, 200)




height, width = image.shape
num_pixels = height * width  # Total number of pixels

for x in range(width):    

    last_black_y = 0
    for y in range(height):  # Process all columns (LEFT MOVEMENT ALLOWED)
        #print(f"y: {y}, x:{x} {image[y,x]} last black x: {last_black_x}")
        if int(image[y,x]) < 255:
            if y - last_black_y > 1:
                pix_val_inciment = 255 / (y - last_black_y)
                pix_value = 254
                for shader in range(last_black_y, y-2):

                    image[shader,x] = pix_value
                    pix_value = pix_value - pix_val_inciment
            last_black_y = y
            

cv2.imwrite(r'C:\Users\User\Documents\PythonProjects\E2CR\debug\shaded.tiff',np.bitwise_not(image))
cv2.imshow("BaseImage",np.bitwise_not(image))
cv2.waitKey()





