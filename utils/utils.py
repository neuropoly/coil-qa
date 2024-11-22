# This file contains utility functions for data processing and visualization.
# 
# Author: Julien Cohen-Adad

import numpy as np
import matplotlib.pyplot as plt


def compute_noise_stats(noise):
    """
    Calculate noise spectrum and return effective noise bandwidth.
    
    Parameters:
    - noise: ndarray
        Input noise data array. Assumes channels are stored in the third dimension.

    Returns:
    - noise_corr: ndarray
        Noise correlation matrix.
    - mean_noise_corr_upper_scaled: float
        Mean of the upper triangle of the correlation matrix, scaled.
    - noise_cov: ndarray
        Noise covariance matrix.
    
    Reference:
        Kellman P, McVeigh ER. Image reconstruction in SNR units: a general method for SNR measurement. 
        Magn Reson Med. 2005 Dec;54(6):1439-47.
    """
    # Handle dimensions
    dims = noise.shape
    if len(dims) == 2:
        dims = (*dims, 1)
    
    num_channels = dims[2]

    # Reshape to samples x channels
    noise = np.reshape(noise, (np.prod(dims[0:2]), num_channels))
    num_samples = noise.shape[0]

    # Compute the correlation matrix
    # Note: No need to compute the FFT as done in previous implementations, because correlation
    # coefficients are scale-invariant and the Fourier transform preserves the energy of the signal
    # (Parseval's theorem)
    noise_corr = np.corrcoef(noise, rowvar=False)  # Set rowvar=False because variables are columns

    # Extract the upper triangle of the correlation matrix, excluding the main diagonal
    noise_corr_upper = np.triu(noise_corr, k=1)  # k=1 excludes the main diagonal

    # Compute the mean of the absolute values
    mean_noise_corr_upper = np.mean(np.abs(noise_corr_upper.flatten()))

    # Compute the scaling factor
    scaling_factor = num_channels**2 / (0.5 * num_channels**2 - num_channels)

    # Compute the scaled mean
    mean_noise_corr_upper_scaled = mean_noise_corr_upper * scaling_factor

    # Compute noise covariance matrix
    noise_cov = np.dot(noise.T.conj(), noise) / (num_samples - 1)

    return noise_corr, mean_noise_corr_upper_scaled, noise_cov


def reconstruct_coil_images(kspace):
    """
    Reconstruct coil sensitivity images from k-space data.

    Parameters:
        kspace (numpy.ndarray): K-space data of shape (Nx, Ny, Nchannels).

    Returns:
        coil_images (numpy.ndarray): Reconstructed coil images of shape (Nx, Ny, Nchannels).
    """
    # Apply inverse FFT along the first two axes (spatial dimensions)
    coil_images = np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(kspace, axes=(0, 1)), 
            axes=(0, 1)),
        axes=(0))
    return coil_images


def combine_rss(coil_images):
    """
    Combine coil images using the root-sum-of-squares (RSS) method.

    Parameters:
        coil_images (numpy.ndarray): Coil images of shape (Nx, Ny, Nchannels).

    Returns:
        rss_image (numpy.ndarray): Combined RSS image of shape (Nx, Ny).
    """
    rss_image = np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=2))
    return rss_image


def calculate_snr_rss(rss_image, noise_cov):
    """
    Calculate SNR maps for the RSS combination.

    Parameters:
        rss_image (numpy.ndarray): Combined RSS image of shape (Nx, Ny).
        noise_cov (numpy.ndarray): Noise covariance matrix of shape (Nchannels, Nchannels).

    Returns:
        snr_map (numpy.ndarray): SNR map of shape (Nx, Ny).
    """
    # Noise variance is the sum of the diagonal elements of the covariance matrix
    noise_variance = np.trace(noise_cov)
    noise_std = np.sqrt(noise_variance)

    # Calculate SNR
    snr_map = rss_image / noise_std
    return snr_map
