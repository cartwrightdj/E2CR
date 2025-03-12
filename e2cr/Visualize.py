import cv2
import numpy as np

def visualize_cc(image, minarea=0):
    # Check if the image is already grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    height, width = image.shape[:2]

    _, image_thresh = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY_INV)

    # Connected component analysis
    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(image_thresh)
    
    # Create colored labels image with 3 channels for RGB
    colored_labels_im = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Generate random colors for each label
    colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)

    # Iterate over labels
    label_stats = []
    for label in range(1, num_labels):  # Start from 1 to skip the background
        x, y, w, h, area = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], \
                           stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT], \
                           stats[label, cv2.CC_STAT_AREA]
        
        component_mask = (labels_im == label).astype(np.uint8)
              
        if area > minarea:
            # Color the pixels of the current label
            
            mask = (labels_im == label)
            colored_labels_im[mask] = colors[label]

            # Add label number to the image
            B, G, R = int(colors[label][0]), int(colors[label][1]), int(colors[label][2])
            if h > 20 or w > 20:
                colored_labels_im = cv2.putText(colored_labels_im, f"{label} {area} (x:{x},y:{y})", 
                                            (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (B, G, R), 1)
            
    return colored_labels_im