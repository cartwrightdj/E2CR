import matplotlib.pyplot as plt

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
    plt.plot(numbers, linestyle='-', color='b')
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


def plot_histogram_with_peaks(data, peaks, threshold, save_path=None):
    """
    Plot the data with identified peaks and a threshold line.
    
    Args:
        data (list or np.array): The input data array.
        peaks (list): Indices of the peaks.
        threshold (float): The threshold value.
        save_path (str): Path to save the plot image. If None, the plot is displayed instead.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(data, marker='o', linestyle='-', color='b', label='Data')
    plt.axhline(y=threshold, color='r', linestyle='--', linewidth=1, label=f'Threshold ({threshold})')
    plt.scatter(peaks, data[peaks], color='green', s=100, zorder=3, label='Peaks above threshold')
    
    for peak in peaks:
        plt.axvline(x=peak, color='g', linestyle=':', linewidth=1)
        plt.annotate(f"{peak}", (peak, data[peak]), textcoords="offset points", xytext=(0,10), ha='center', color='green')
    
    plt.title('Data with Peaks Above Threshold')
    plt.xlabel('Index')
    plt.ylabel('Value')
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