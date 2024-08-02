import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import cv2

def gaussian(x, mean, amp, stddev): 
    """
    Compute the value of a Gaussian function.

    Parameters:
    x (float or numpy.ndarray): The input value(s) at which to evaluate the Gaussian
    mean (float): The mean of the Gaussian
    amp (float): The amplitude of the Gaussian
    stddev (float): The standard deviation of the Gaussian

    Returns: float or numpy.ndarray: The value(s) of the Gaussian function at the input value(s) x.
    """

    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def fit_gaussian(data, bins=100): # Bins default is 100
    """
    Fit a Gaussian to the histogram of given divided image data

    Parameters:
    data (2d array): 1400x1400 pixel values post division to be saved in histogram and fitted
    bins (int): # of bins for histogram

    Returns:
    A tuple containing:
        - popt (numpy.ndarray): Optimal values for the parameters of the Gaussian 
        - bin_centers (numpy.ndarray): The centers of the bins used in the histogram
        - hist (numpy.ndarray): The counts of the data in each histogram bin
        - fitted_values (numpy.ndarray): The values of the fitted Gaussian at the bin centers
    """

    # Generate the histogram data
    hist, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fit the Gaussian to the histogram data
    popt, pcov = curve_fit(gaussian, bin_centers, hist, p0=[np.mean(data), np.max(hist), np.std(data)])
    fitted_values = gaussian(bin_centers, *popt)
    
    return popt, bin_centers, hist, fitted_values

def calculate_mse(observed, predicted):
    """
    Calculate the Mean Squared Error (MSE) between the observed and predicted values.

    Parameters:
    observed (numpy.ndarray): The observed data
    predicted (numpy.ndarray): The predicted data from the Gaussian

    Returns: MSE as float 
    """
    
    return np.mean((observed - predicted) ** 2)

def calculate_normalized_mse(observed, predicted):
    """
    Calculate the Normalized Mean Squared Error (NMSE) between the observed and predicted values.
    (MSE / variance of observed data)

    Parameters:
    observed (numpy.ndarray): Observed data
    predicted (numpy.ndarray): Predicted data from the Gaussian

    Returns: NMSE as float
    """

    mse = calculate_mse(observed, predicted)
    return mse / np.var(observed)
    
# Adjust max_iterations, threshold_factor and mse_threshold as necessary
def iterative_gaussian_fit(data, output_dir, file_name, max_iterations=20, nmse_threshold = 0.001):
    """
    Iteratively fits a Gaussian to the data, trimming high-value pixels to improve the fit.

    Parameters:
    data (array): 196,000 pixel values after division and flattening
    output_dir (str): Directory where output file will be stored, same output_dir as divide_images
    file_name (str): Name of the outputted plot, same file_name as divide_images
    max_iterations (int, optional): The maximum number of iterations for refitting, default is 20
    Adjust based on the left and right thresholds printed.
    nmse_threshold (float, optional): Threshold to determine how much nmse has to decrease to stop iterating,
    default is 0.0002

    Returns: None
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Tracking previous NMSE to see when to terminate loop
    prev_nmse = 0
    # Saving variables for normalization later 
    # data_min = np.min(data)
    # data_max = np.max(data)
    # Looping and refitting trimmed data to decrease NMSE
    for i in range(max_iterations):
        # Fit the data and retrieve important data
        popt, bin_centers, hist, fitted_values = fit_gaussian(data)
        # Calculate MSE and NMSE and print it out
        mse = calculate_mse(hist, fitted_values)
        nmse = calculate_normalized_mse(hist, fitted_values)
        print(f'Iteration {i}')
        print(f'MSE: {mse}')
        print(f'Normalized MSE: {nmse}')
        # Remove high-value pixels from the right and left side of the curve one standard deviation at a time
        right_threshold = np.max(data) - np.std(data)
        left_threshold = 2*np.mean(data) - right_threshold
        print(f"right: {right_threshold}")
        print(f"left: {left_threshold}")
        print(f"size before: {data.size}")
        # Uncomment next 5 lines if you want to see if # of iterations is enough (check saved image)
        # To see if the bright and dark spots have disappeared
        #  data = np.reshape(data,(1400,1400))
        # for i in range(1400): 
        #     for j in range(1400): 
        #         if data[i][j] > right_threshold: data[i][j] = np.mean(data)
        #         if data[i][j] < left_threshold: data[i][j] = np.mean(data)
        data = data[(data > left_threshold) & (data < right_threshold)]
        # Uncomment next 6 lines to check saved image
        # output = data
        # output = (output - data_min) / (data_max - data_min)
        # output *= 65535
        # # Convert from float to int
        # output = np.array(output, dtype=np.uint16)
        # cv2.imwrite(os.path.join(output_dir, f'trim_{file_name}'), output)
        # data = data.flatten()
        print(f"size after: {data.size}")

    # Plot the final gaussian and dataset
    fig, ax = plt.subplots()
    ax.hist(data, bins=len(bin_centers), density=False, alpha=0.6, color='g', label='Observed')
    ax.plot(bin_centers, fitted_values, 'r--', label='Fitted Gaussian')
    # Extract mean and standard deviation of Gaussian
    m = popt[0]; std = popt[2]
    ax.axvline(m, color='k', linestyle='dashed', label='Mean')
    ax.axvline(m + std, color='y', linestyle='dashed', label='Mean + Std Dev')
    ax.axvline(m - std, color='y', linestyle='dashed', label='Mean - Std Dev')
    ax.set_title(f'Normalized MSE: {nmse: .7f} Mean: {m: .5f} Std: {std: .6f}')
    ax.legend()
    plt.show()
    # Uncomment next two lines to save the histogram to file directory
    # fig.savefig(os.path.join(output_dir, f'finalfit_{file_name}'))
    # plt.close(fig)