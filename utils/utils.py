# This file contains utility functions for data processing and visualization.
# A lot of these functions were originally written by Jon Polimeni and converted to Python by Julien Cohen-Adad.

import numpy as np

import numpy as np
import matplotlib.pyplot as plt

def compute_noise_bandwidth(noise, display=False):
    """
    Calculate noise spectrum and return effective noise bandwidth.
    
    Parameters:
    - noise: ndarray
        Input noise data array. Assumes channels are stored in the third dimension.
    - display: bool, optional
        Whether to display the noise power spectrum plot (default: False).
    
    Returns:
    - noise_bandwidth: float
        Effective noise bandwidth.
    - noise_bandwidth_chan: ndarray
        Noise bandwidth for each channel.
    - N_power_spectrum_avg: ndarray
        Average noise power spectrum.
    """
    # Ensure dimensions are handled correctly
    dims = list(noise.shape)
    while len(dims) < 16:
        dims.append(1)

    # Reshape to samples x channels
    noise = np.transpose(noise, axes=list(range(len(dims) - 1)) + [len(dims) - 1])
    noise = noise.reshape(-1, dims[2])

    # Compute FFT
    N = np.fft.fft(noise.reshape(dims[0], -1, dims[2]), axis=0) / np.sqrt(dims[0])
    N_power_spectrum = np.abs(N) ** 2

    # Channel-wise power spectrum
    N_power_spectrum_chan = np.mean(N_power_spectrum, axis=1)
    N_power_spectrum_chan_normalized = N_power_spectrum_chan / N_power_spectrum_chan[0, :]
    noise_bandwidth_chan = np.mean(N_power_spectrum_chan_normalized, axis=0)

    # Average power spectrum
    N_power_spectrum_avg = np.mean(N_power_spectrum_chan, axis=1)
    N_power_spectrum_avg_normalized = N_power_spectrum_avg / N_power_spectrum_avg[0]
    noise_bandwidth = np.mean(N_power_spectrum_avg_normalized)

    # Display plot if required
    if display:
        plt.figure()
        plt.gca().add_patch(plt.Rectangle((0.25 * dims[0], 0.001), 0.5 * dims[0], 1.098, 
                                          facecolor='0.9', linestyle='none'))
        plt.plot(np.fft.fftshift(N_power_spectrum_avg_normalized))
        plt.xticks(
            [0, dims[0] * 0.25, dims[0] * 0.5, dims[0] * 0.75, dims[0]],
            [-0.5, -0.25, 0, 0.25, 0.5]
        )
        plt.xlim([0, dims[0]])
        plt.ylim([0, 1.1])
        plt.grid(axis='y')
        plt.xlabel('Frequency (normalized)')
        plt.ylabel('Power (DC normalized)')
        plt.title(f'Average of normalized noise power spectrum, BW={noise_bandwidth:.3f}')
        plt.box(on=True)
        plt.show()

    return noise_bandwidth, noise_bandwidth_chan, N_power_spectrum_avg


def array_stats_matrix(rawdata, stat_type='cov', do_noise_bw_scaling=False):
    """
    Computes various statistical matrices for multi-channel array data.
    
    Parameters:
    - rawdata: ndarray
        The raw data array. The third dimension corresponds to coil channels.
    - stat_type: str
        Type of statistical matrix to compute:
        'cor'  - Correlation matrix
        'cof'  - Correlation coefficient matrix
        'cov'  - Covariance matrix (default)
        'std'  - Standard deviation matrix
        'avg'  - Average vector
    - do_noise_bw_scaling: bool
        Whether to scale the noise bandwidth.
    
    Returns:
    - result: ndarray
        The computed statistical matrix or vector.
    - noise (optional): ndarray
        The reshaped noise data if requested.
    
    Code inspired by Jon Polimeni
    """
    # Ensure data is double precision
    noise = rawdata.astype(np.float64)

    if do_noise_bw_scaling:
        noise_bandwidth = compute_noise_bandwidth(noise)
        if noise_bandwidth < 0.6:
            print("Warning: Noise bandwidth is too low; data may not be pure noise.")

    # Handle dimensions
    dims = noise.shape
    if len(dims) == 2:
        dims = (*dims, 1)

    # Permute dimensions to move the coil channel to the last dimension
    noise = np.transpose(noise, axes=list(range(len(dims) - 1)) + [len(dims) - 1])

    # Reshape into samples X channels
    noise = noise.reshape(-1, dims[2])

    # Transpose so each observation is a column vector
    noise = noise.T

    # Dimensions
    Nchan, Nsamp = noise.shape

    if np.count_nonzero(noise) != noise.size:
        print(f"Warning: {100 * (1 - np.count_nonzero(noise) / noise.size):.1f}% of noise data elements are zero-valued.")

    # Compute the requested statistic
    if stat_type.lower() == 'mtx':
        result = noise
    elif stat_type.lower() == 'cor':
        cormtx = (noise @ noise.T) / (Nsamp - 1)
        if do_noise_bw_scaling:
            print(f"\nScaling covariance matrix by effective noise bandwidth: {noise_bandwidth:.4f}")
            cormtx /= noise_bandwidth
        result = cormtx
    elif stat_type.lower() == 'cov':
        avgvec = np.mean(noise, axis=1, keepdims=True)
        covmtx = (noise @ noise.T - avgvec @ avgvec.T * Nsamp) / (Nsamp - 1)
        if do_noise_bw_scaling:
            print(f"\nScaling covariance matrix by effective noise bandwidth: {noise_bandwidth:.4f}")
            covmtx /= noise_bandwidth
        result = covmtx
    elif stat_type.lower() == 'cof':
        avgvec = np.mean(noise, axis=1, keepdims=True)
        covmtx = (noise @ noise.T - avgvec @ avgvec.T * Nsamp) / (Nsamp - 1)
        stdvec = np.sqrt(np.diag(covmtx))
        result = (covmtx / np.outer(stdvec, stdvec))
    elif stat_type.lower() == 'std':
        avgvec = np.mean(noise, axis=1)
        covmtx = (noise @ noise.T - avgvec @ avgvec.T * Nsamp) / (Nsamp - 1)
        stdvec = np.sqrt(np.diag(covmtx))
        stdvec = stdvec / np.sum(stdvec) * Nchan
        result = stdvec
    elif stat_type.lower() in ['mean', 'avg']:
        result = np.mean(noise, axis=1)
    elif stat_type.lower() == 'avgmtx':
        avgvec = np.mean(noise, axis=1, keepdims=True)
        result = avgvec @ avgvec.T
    elif stat_type.lower() == 'res':
        result = (noise @ np.linalg.pinv(noise)) / (Nsamp - 1)
    elif stat_type.lower() == 'onc':
        result = sum(noise[:, i:i+1] @ np.linalg.pinv(noise[:, i:i+1]) for i in range(Nsamp))
    else:
        raise ValueError(f"Unknown statistics matrix type: {stat_type}")

    return result, noise if do_noise_bw_scaling else result
