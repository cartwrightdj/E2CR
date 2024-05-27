'''Example:
Script Name: utils.py /.utils.py

Author: David J. Cartwright]
Date: 5/25/2024
Last Modification Date:

Dependencies:
    - os (standard library)
'''
import os

def create_unique_filepath(path):
    """
    Ensure necessary folders are created for the given path. If it's a file path,
    handle the situation where the file already exists by appending a number at the end.

    Args:
        path (str): The desired file or folder path.

    Returns:
        str: The unique file path with necessary folders created.
    """
    # Check if the path is a file or a directory
    base, extension = os.path.splitext(path)
    if extension:  # It's a file path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # Check if file already exists and create a unique file path
        counter = 1
        new_filepath = path
        while os.path.exists(new_filepath):
            new_filepath = f"{base}_{counter}{extension}"
            counter += 1
        
        return new_filepath
    else:  # It's a directory path
        os.makedirs(path, exist_ok=True)
        return path