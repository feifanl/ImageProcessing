import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import cv2
from align_images import align_images
from gaussian_fit import iterative_gaussian_fit

def average_images(data, output_dir, name): 
    """
    Average 100 1400x1400 images. 

    Parameters: 
    data (3d array): 100x1400x1400 array which contains all images
    output_dir(str): Directory to save averaged image to 
    name (str): Number sequence at the end of "Process_", ex. 64206

    Returns: average (2d array): 1400x1400 array containing averaged pixel values of all 100 images
    """

    # Makes directory if does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Average image data
    average = data.mean(axis = 0)
    # Convert to uint16
    average = np.array(average, dtype=np.uint16)
    # Creates image
    cv2.imwrite(os.path.join(output_dir, f'{name}_average.png'), average)
    return average

def create_histogram(input_dir, num_bins, output_dir, file_name, data):  
    """
    Saves given image data or reads it from an image as a histogram displaying pixel values.

    Parameters: 
    input_dir (str): Directory to read the image from if data is not provided
    num_bins (int): # of bins for the histogram
    output_dir (str): Directory where the histogram is saved
    file_name (str): File name for the saved histogram
    data (2d array, optional): 1400x1400 array containing image data. If not provided, the image will be 
    read from input_dir

    Returns: None
    """
    
    # Checking if data has been given, if not, read the image from input_dir
    if not hasattr(data, "__len__"): 
        # Reading images
        data = cv2.imread(input_dir, cv2.IMREAD_ANYDEPTH)
    # Checking if directories exist, if not, create them
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    fig, ax = plt.subplots()
    # Flatten data to 1d
    ax.hist(data.flatten(), bins=num_bins)
    m = data.mean()
    std = data.std()
    ax.axvline(m, color='k', linestyle='dashed')
    ax.axvline(m + std, color='y', linestyle='dashed')
    ax.axvline(m - std, color='y', linestyle='dashed')
    ax.set_title(f'Mean: {m}, Std Dev: {std}')
    fig.savefig(os.path.join(output_dir, file_name))
    plt.close(fig)

def divide_images(image1, image2, output_dir, file_name): 
    """
    Takes two images, aligns them and crops them to remove blank values, divides image2 by image1 and saves
    the image result, histogram displaying the pixel values after divison, and removes high value pixels to 
    fit a histogram with a gaussian to measure noise. 

    Parameters:
    image1 (2d array): 1400x1400 array containing pixel values
    image2 (2d array): 1400x1400 array containing pixel values
    output_dir (str): Directory to save all outputted images/histograms
    file_name (str): File name, ex. 64206_divby_64528

    Returns: None
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Resize using bilinear interpolation for better divison
    image1 = cv2.resize(image1, (4000,4000))
    image2 = cv2.resize(image2, (4000,4000))
    # Aligning
    image1 = align_images(image1, image2, 10000, output_dir, file_name)
    # Trim zeros 
    non_zero_rows = np.any(image1 != 0, axis=1)
    non_zero_cols = np.any(image1 != 0, axis=0)
    image1 = image1[non_zero_rows][:, non_zero_cols]
    image2 = image2[non_zero_rows][:, non_zero_cols]
    # Crop the images to the smallest dimension to make them square
    min_dim = min(image1.shape[0], image1.shape[1])
    image1 = image1[:min_dim, :min_dim]
    image2 = image2[:min_dim, :min_dim]
    # Divide images
    output = image2 / image1
    # Scale back down to 1400 x 1400
    output = cv2.resize(output, (1400, 1400))
    # Create histograms and eliminate outliers to maximize GOF (measured by NMSE)
    create_histogram("", 20, output_dir, f"hist_{file_name}", output)
    iterative_gaussian_fit(output.flatten(), output_dir, file_name)
    # Normalizing data and scaling up to uint16
    output = (output - np.min(output)) / (np.max(output) - np.min(output))
    output *= 65535
    # Convert from float to int
    image_data = np.array(output, dtype=np.uint16)
    cv2.imwrite(os.path.join(output_dir, file_name), image_data)