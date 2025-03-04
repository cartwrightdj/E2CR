
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Document:
    def __init__(self,image_path) -> None:
        image = cv2.imread(image_path)
        if not image is None:
            self._original = image.copy()
            if len(self._original.shape) == 3:
                self._grayscale = cv2.cvtColor(self._original, cv2.COLOR_BGR2GRAY)
                self._grayscale = 255 - self._grayscale
                self._init_grayscale()
        else:
            Warning(f"Could not loaf image: {image_path}")
    
    def show_original(self):
        cv2.imshow("original",self._original)
        cv2.waitKey()

    @property
    def origonal_shape(self):
        return self._original.shape
    
    def _init_grayscale(self):       
        self.grayscale_h_hist = np.sum(self._original, axis=1)  # Sum along the columns (for rows)
        self.grayscale_v_hist = np.sum(self._original, axis=0)    # Sum along the rows (for columns)

    def _calculate_contrast(self):
        self._std_dev = np.std(self._grayscale)

        I_max = np.max(self._grayscale)
        I_min = np.min(self._grayscale)
        self._michelson_contrast = (I_max - I_min) / (I_max + I_min)

        mean_intensity = np.mean(self._grayscale)
        self._rms_contrast = np.sqrt(np.mean((self._grayscale - mean_intensity) ** 2))

        print(f"StdDev: {self._std_dev}, michelson: {self._michelson_contrast}, rms: {self._rms_contrast}")


    def plot_grayscale_shit(self):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

        # Display the original image
        ax1.imshow(self._grayscale, cmap='gray')
        ax1.set_title('Original Image')

        # Plot the horizontal histogram
        ax2.plot(self.grayscale_h_hist, np.arange(len(self.grayscale_h_hist)))
        ax2.invert_yaxis()
        ax2.set_title('Horizontal Histogram')

        # Plot the vertical histogram
        ax3.plot(self.grayscale_v_hist)
        ax3.set_title('Vertical Histogram')

        # Show the plots
        plt.tight_layout()
        plt.show()


doc = Document("C:/Users/User/Documents/E2CR/sample_images_for_ocr/4159363_00362.jpg")
doc.plot_grayscale_shit()
doc._calculate_contrast()


top_y_border = None
bottom_y_border = None
left_x_border = None
right_x_border = None

# Loop through the rows of the image
for i in range(doc._grayscale.shape[0]):  # Loop over rows
    row_sum = np.sum(doc._grayscale[i, :])  # Sum the pixel values in the row
    
    # Find the first row where sum is below the threshold
    
    if top_y_border is None and (row_sum) / (doc._grayscale.shape[1] * 255) < .5:
        print((row_sum) / (doc._grayscale.shape[1] * 255))
        top_y_border = i
    
    # Find the next row where sum exceeds the threshold
    if top_y_border is not None and (row_sum) / (doc._grayscale.shape[1] * 255) > .5:
        print((row_sum) / (doc._grayscale.shape[1] * 255))
        bottom_y_border = i
        break  # Exit the loop after finding the second row

# Loop through the rows of the image
for i in range(doc._grayscale.shape[1]):  # Loop over rows
    row_sum = np.sum(doc._grayscale[:,i])  # Sum the pixel values in the row
    
    # Find the first row where sum is below the threshold
    
    if left_x_border is None and (row_sum) / (doc._grayscale.shape[0] * 255) < .5:
        print((row_sum) / (doc._grayscale.shape[0] * 255))
        left_x_border = i
    
    # Find the next row where sum exceeds the threshold
    if left_x_border is not None and (row_sum) / (doc._grayscale.shape[0] * 255) > .5:
        print((row_sum) / (doc._grayscale.shape[0] * 255))
        right_x_border = i
        print(f"{right_x_border=}")
        break  # Exit the loop after finding the second row


row_means = np.mean(doc._grayscale, axis=1)

# Calculate the average of the row means
average_of_rows = np.mean(row_means)

print(f'Average value of all rows: {average_of_rows/255}')



y_coordinate = 100  # The row where the line should be drawn
start_point = (0, top_y_border)  # Starting at the left of the image
start_point2 = (0, bottom_y_border)  # Starting at the left of the image
start_point3 = (left_x_border,0)  # Starting at the left of the image
start_point4 = (right_x_border, 0)  # Starting at the left of the image
end_point = (doc._original.shape[1], top_y_border)  # Ending at the right of the image
end_point2 = (doc._original.shape[1], bottom_y_border)  # Ending at the right of the image
end_point3 = (left_x_border,doc._original.shape[0])  # Ending at the right of the image
end_point4 = (right_x_border,doc._original.shape[0])  # Ending at the right of the image

# Draw the line on the image
green = (0, 255, 0)  # Line color in BGR (Green in this case)
red = (0,0,255)
thickness = 2  # Thickness of the line
cv2.line(doc._original, start_point, end_point, green, thickness)
cv2.line(doc._original, start_point2, end_point2, red, thickness)
cv2.line(doc._original, start_point3, end_point3, green, thickness)
cv2.line(doc._original, start_point4, end_point4, red, thickness)

# Show the image with the line
cv2.imshow('Image with Horizontal Line', doc._original)

# Wait for a key press and close the displayed image window
cv2.waitKey(0)
cv2.destroyAllWindows()