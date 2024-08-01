import numpy as np
import os
import cv2

def process_file(file_pattern):
    """
    Takes a 392 MB ets file containing 100 images and reads them, stores into a 3d array.

    Parameters:
    file_pattern (str): ets file directory to read from

    Returns: data (3d array): 100x1400x1400 np array containing all image data
    """

    # Creating 100x1400x1400 array filled with zeros, datatype uint16 
    data = np.zeros((100, 1400, 1400), dtype=np.uint16)
    # Filling the data array with the data from the ets file 
    with open(file_pattern, 'rb') as file:
        for i in range(100):
            # Skip the header (292 bytes)
            file.seek(292 + (3920000 * i)) 
            # Fills 2d arrays at index i 
            data[i, :, :] = np.fromfile(file, dtype=np.uint16, count=1400 * 1400).reshape((1400, 1400))
        file.close()
    return data

def save_images(data, output_dir, name):
    """
    Saves all 100 images to a given file directory.

    Parameters:
    data (3d array): 100x1400x1400 array containing all images and their pixel values
    output_dir (str): Directory to save the images to
    name (str): Name of the file that the images come from, ex. "64206"

    Returns: None
    """

    # Creates directory if does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Creates 100 images by taking corresponding 2d array from data array
    for i in range(100):
        cv2.imwrite(os.path.join(output_dir, f'{name}_image_{i}.png'), data[i, :, :])
