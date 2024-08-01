import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

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
def iterative_gaussian_fit(data, output_dir, file_name, max_iterations=5, threshold_factor=5, mse_threshold=1e-10):
    """
    Iteratively fits a Gaussian to the data, trimming high-value pixels to improve the fit.

    Parameters:
    data (array): 196,000 pixel values after division and flattening
    output_dir (str): Directory where output file will be stored, same output_dir as divide_images
    file_name (str): Name of the outputted plot, same file_name as divide_images
    max_iterations (int, optional): The maximum number of iterations for refitting, default is 5
    threshold_factor (float, optional): Factor to adjust trim of high-value pixels, default is 5
    mse_threshold (float, optional): Threshold to determining improvement in NMSE, default is 1e-10

    Returns: None
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Setting NMSE as high as possible to compare with initial iteration
    previous_nmse = float('inf')
    # Looping and refitting trimmed data to decrease NMSE
    for i in range(max_iterations):
        # Fit the data and retrieve important data
        popt, bin_centers, hist, fitted_values = fit_gaussian(data)
        # Calculate MSE and NMSE and print it out
        mse = calculate_mse(hist, fitted_values)
        normalized_mse = calculate_normalized_mse(hist, fitted_values)
        print(f'Iteration {i}')
        print(f'MSE: {mse}')
        print(f'Normalized MSE: {normalized_mse}')
        # If NMSE is not improving significantly (based on mse_threshold), terminate loop
        if previous_nmse - normalized_mse < mse_threshold:
            print('Stopping as NMSE did not improve significantly.')
            break
        # Set previous nmse as the current one to compare to next iteration
        previous_nmse = normalized_mse
        # Remove high-value pixels from the right side of the curve
        threshold = np.mean(data) + threshold_factor * np.std(data)
        data = data[data < threshold]
    
    # Plot the final gaussian and dataset
    fig, ax = plt.subplots()
    ax.hist(data, bins=len(bin_centers), density=False, alpha=0.6, color='g', label='Observed')
    ax.plot(bin_centers, fitted_values, 'r--', label='Fitted Gaussian')
    # Extract mean and standard deviation of Gaussian
    m = popt[0]; std = popt[2]
    ax.axvline(m, color='k', linestyle='dashed', label='Mean')
    ax.axvline(m + std, color='y', linestyle='dashed', label='Mean + Std Dev')
    ax.axvline(m - std, color='y', linestyle='dashed', label='Mean - Std Dev')
    ax.set_title(f'Normalized MSE: {normalized_mse: .7f} Mean: {m: .5f} Std: {std: .6f}')
    ax.legend()
    fig.savefig(os.path.join(output_dir, f'finalfit_{file_name}'))
    plt.close(fig)