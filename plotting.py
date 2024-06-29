import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

def plot_numbers(numbers, title="Image Histogram", x_label="Index", y_label="Value", marked_x_values=None, threshold_value=None, save_path=None):
    """
    Plot a list of numbers using matplotlib, with the ability to mark certain x-values.
    
    Args:
        numbers (list): List of numerical values to plot.
        title (str): Title of the plot.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        marked_x_values (list): List of x-values to mark on the plot.
        save_path (str): Path to save the plot image. If None, the plot is not saved.
    """
    plt.figure(figsize=(10, 6))
    plt.barh(numbers, linestyle='-', color='b',width=200)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    
    #if marked_x_values:
    #    for x in marked_x_values:
    #        if 0 <= x < len(numbers):
    #            plt.plot(x, numbers[x], 'ro')  # 'ro' means red color with circle marker
     #           plt.annotate(f"{numbers[x]}", (x, numbers[x]), textcoords="offset points", xytext=(0,10), ha='center')

    if not threshold_value is None:
        for y in threshold_value:
            plt.axhline(y=y, color='r', linestyle='--')  # Draw a horizontal line at the y-value
            plt.annotate(f"{y}", (0, y), textcoords="offset points", xytext=(0, 10), ha='left', color='red')

    if not marked_x_values is None:
        for x in marked_x_values:
            if 0 <= x < len(numbers):
                y = numbers[x]
                plt.axvline(x=x, color='g', linestyle=':')  # Draw a vertical line at the x-value
                


    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_histogram_with_peaks(data, peaks, threshold, save_path=None, horizontal=True):
    """
    Plot the data with identified peaks and a threshold line.
    
    Args:
        data (list or np.array): The input data array.
        peaks (list): Indices of the peaks.
        threshold (float): The threshold value.
        save_path (str): Path to save the plot image. If None, the plot is displayed instead.
        horizontal (bool): If True, plot the data horizontally.
    """
    plt.figure(figsize=(12, 8))

    data = normalize_list(data)
    
    if horizontal:
        plt.plot(data, range(len(data)), linestyle='-', color='b', label='Data')
        plt.axvline(x=threshold, color='r', linestyle='--', linewidth=1, label=f'Threshold ({threshold})')
        plt.scatter(np.array(data)[peaks], peaks, color='green', s=100, zorder=3, label='Peaks above threshold')
        
        for peak in peaks:
            plt.axhline(y=peak, color='g', linestyle=':', linewidth=1)
            plt.annotate(f"{peak}", (data[peak], peak), textcoords="offset points", xytext=(0,10), ha='center', color='green')
            
        plt.xlabel('Normalized Value')
        plt.ylabel('Index')
    else:
        plt.plot(range(len(data)), data, linestyle='-', color='b', label='Data')
        plt.axhline(y=threshold, color='r', linestyle='--', linewidth=1, label=f'Threshold ({threshold})')
        plt.scatter(peaks, np.array(data)[peaks], color='green', s=100, zorder=3, label='Peaks above threshold')
        
        for peak in peaks:
            plt.axvline(x=peak, color='g', linestyle=':', linewidth=1)
            plt.annotate(f"{peak}", (peak, data[peak]), textcoords="offset points", xytext=(0,10), ha='center', color='green')
        
        plt.xlabel('Index')
        plt.ylabel('Normalized Value')

    plt.title('Data with Peaks Above Threshold')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_high_points(data, peaks, threshold, save_path=None):
    """
    Plot the data with identified high points (peaks) and a threshold line.
    
    Args:
        data (list or np.array): The input data array.
        peaks (list): Indices of the high points (peaks).
        threshold (float): The threshold value.
        save_path (str): Path to save the plot image. If None, the plot is displayed instead.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(data, marker='o', linestyle='-', color='b', label='Data')
    plt.axhline(y=threshold, color='r', linestyle='--', linewidth=1, label=f'Threshold ({threshold})')
    plt.scatter(peaks, data[peaks], color='green', s=100, zorder=3, label='High Points above threshold')
    
    for peak in peaks:
        plt.axvline(x=peak, color='g', linestyle=':', linewidth=1)
        plt.annotate(f"{peak}", (peak, data[peak]), textcoords="offset points", xytext=(0,10), ha='center', color='green')
    
    plt.title('Data with High Points Above Threshold')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_clusters(data, labels, title="Data Grouped by Proximity", save_path=None):
    """
    Plot data with clusters highlighted.
    
    Args:
        data (list or np.array): List of numerical values to plot.
        labels (list or np.array): Cluster labels for each point.
        title (str): Title of the plot.
        save_path (str): Path to save the plot image. If None, the plot is not saved.
    """
    plt.figure(figsize=(12, 8))
    plt.scatter(range(len(data)), data, c=labels, cmap='viridis', s=100)
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_horizontal_histogram(values, bins=50, title='Horizontal Histogram', xlabel='Frequency', ylabel='Value', show=True):
    """
    Plots a horizontal histogram for the given values.

    Parameters:
    values (np.ndarray): The input values for the histogram.
    bins (int): The number of bins for the histogram.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    show (bool): If True, display the plot. If False, return the figure and axis objects.

    Returns:
    (plt.Figure, plt.Axes): The figure and axis objects (if show is False).
    """
    

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(values, align='center')

    # Set titles and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Show or return the plot
    if show:
        plt.show()
    else:
        return fig, ax
    
def plot_horizontal_first_contact(first_contact):
    """
    Plots the first contact points horizontally.

    Parameters:
    image (np.ndarray): The input image.
    threshold (int): The pixel value threshold for detecting contact points.
    """
    
   
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(first_contact, range(len(first_contact)), 'o-', label='First Contact Points')

    ax.set_title('First Contact Points per Row')
    ax.set_xlabel('Column Index of First Contact')
    ax.set_ylabel('Row Index')
    ax.invert_yaxis()  # To have the origin (0,0) at the top-left corner
    ax.legend()

    plt.show()

def plot2(data):
    print(data)
    pdata = data
    data = np.array(data)
    try:
        mode_result = stats.mode(data)
        print(f"mode_results: {mode_result}")
        mode_val = mode_result.mode[0]
    except IndexError:
        # If mode_result.mode[0] fails, calculate mode using histogram
        print("IndexError")
        counts, bin_edges = np.histogram(data, bins=30)
        mode_bin_index = np.argmax(counts)
        mode_val = (bin_edges[mode_bin_index] + bin_edges[mode_bin_index + 1]) / 2
   

    print(f"mode_val: {mode_val}")
    # Create the plot
    plt.barh(pdata,align='center',width=5)
    plt.title("Histogram with Mode Highlighted")
    plt.ylabel("Value")
    plt.xlabel("Frequency")
    
    # Add a vertical line at the mode
    plt.axhline(mode_val, color='red', linestyle='dashed', linewidth=1, label=f'Mode: {mode_val}')
    
    # Add a legend
    plt.legend()
    
    # Show the plot
    plt.show()

def normalize_list(values):
    """
    Normalize a list of integers to the range [0, 1].

    Args:
        values (list of int): The list of integers to normalize.

    Returns:
        list of float: The normalized list of values.
    """
    min_val = min(values)
    max_val = max(values)

    if min_val == max_val:
        raise ValueError("Normalization is not possible with all equal values")

    #normalized_values = [(v - min_val) / (max_val - min_val) for v in values]
    normalized_values = [(v - min_val)  for v in values]
    return normalized_values

def plot_two_horizontal_projections(norm_projection1, norm_projection2, save_path=None):
    """
    Plot the horizontal projection profiles of two images side by side.
    
    Args:
        image1 (np.ndarray): The first input image.
        image2 (np.ndarray): The second input image.
        save_path (str): Path to save the plot image. If None, the plot is displayed instead.
    """
    # Calculate the horizontal projection profiles
   
    
    # Plot the projections
    plt.figure(figsize=(12, 8))
    
    # Plot the first image projection
    plt.subplot(1, 2, 1)
    plt.plot(norm_projection1, range(len(norm_projection1)), color='b')
    plt.xlabel('Normalized Value')
    plt.ylabel('Row Index')
    plt.title('Horizontal Projection - Image 1')
    plt.grid(True)
    
    # Plot the second image projection
    plt.subplot(1, 2, 2)
    plt.plot(norm_projection2, range(len(norm_projection2)), color='r')
    plt.xlabel('Normalized Value')
    plt.ylabel('Row Index')
    plt.title('Horizontal Projection - Image 2')
    plt.grid(True)
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()