# This script computes SNR, noise correlation and g-factor maps.
# The inputs are:
# 1. Raw data (k-space for each Rx channel) that contains signal from which to compute metrics
# 2. Raw data acquired with the same coil and the same sequence, but without any RF voltage. 
#    This data is used to compute noise correlation and noise covariance matrices.
# 
# The input format can be either:
# - Siemens "meas.dat" files (for raw data and noise data). This requires the custom function "read_meas_dat".
# - GE "p-files". This requires the software Orchestra-SDK.

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys

from utils.utils import compute_noise_stats, array_stats_matrix, mrir_conventional_2d, mrir_array_combine_rss, reconstruct_coil_images, combine_rss, calculate_snr_rss
from utils.ge_utils import process_noise_ge

# Add paths to custom functions (not necessary in Python, assuming functions are available)
# Custom function placeholders:
# read_meas_dat, mrir_ice_dimensions, mrir_array_stats_matrix, mrir_conventional_2d, mrir_array_combine_rss, mrir_array_SNR_rss

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Compute SNR, noise correlation, and g-factor maps.')
    parser.add_argument('fname_image', type=str, help='Path to the raw image data file')
    parser.add_argument('fname_noise', type=str, help='Path to the raw noise data file')
    args = parser.parse_args()

    fname_image = args.fname_image
    fname_noise = args.fname_noise

    # Identify which vendor's data is being used
    if fname_image.endswith('.dat'):
        vendor = 'siemens'
        # TODO: add support for other Siemens file formats
        # Display not covered feature error
        raise NotImplementedError('This feature is not yet implemented')
    elif fname_image.endswith('.7'):
        vendor = 'ge'
    else:
        raise ValueError('Unknown file extension')

    # Read proprietary raw 'image' data
    if vendor == 'siemens':
        meas_image = read_meas_dat(fname_image)
        meas_noise = read_meas_dat(fname_noise)
    elif vendor == 'ge':
        sys.path.append('/Users/julien/code/orchestra-sdk-2.1-1')  # https://github.com/neuropoly/coil-qa/issues/2
        from GERecon import Pfile
        # Read GE 'p-file' data
        pfile = Pfile(fname_image)
        metadata = pfile.MetaData()

        meas_image = pfile.KSpace(0, 0)  # TODO: replace with (nslice, echo)
        pfile = Pfile(fname_noise)
        meas_noise = pfile.KSpace(0, 0)  # TODO: replace with (nslice, echo)
        pass

    # # Display dimensions of image and noise data
    # mrir_ice_dimensions(meas_image['data'])
    # mrir_ice_dimensions(meas_noise['data'])

    # Compute noise statistics
    noise_corr, mean_noise_corr_upper_scaled, noise_cov = compute_noise_stats(meas_noise)
    print(f' ==> average off-diagonal coupling: {mean_noise_corr_upper_scaled * 100:.2f}%')

    # Display noise correlation and covariance matrices
    for matrix, title, fname_suffix in zip([noise_corr, noise_cov], 
                                           ['Noise Correlation Matrix', 'Noise Covariance Matrix'], 
                                           ['noisecorr', 'noisecov']):
        plt.figure()
        plt.imshow(np.abs(matrix), cmap='jet', interpolation='nearest')
        # change label values so that the index starts at 1
        plt.xticks(np.arange(matrix.shape[0]), np.arange(1, matrix.shape[0] + 1))
        plt.yticks(np.arange(matrix.shape[0]), np.arange(1, matrix.shape[0] + 1))
        plt.colorbar()
        plt.title(title)
        plt.savefig(f'{fname_image}_{fname_suffix}.png')

    # Reconstruct coil sensitivity images
    coil_images = reconstruct_coil_images(meas_image)

    # Combine images using RSS
    img_rss = combine_rss(coil_images)

    plt.figure()
    plt.imshow(img_rss, cmap='gray', aspect='equal')
    plt.title('Image After RSS Combination')
    plt.colorbar()
    plt.savefig(f'{fname_image}_rss.png')

    # Calculate SNR map
    snr_rss = calculate_snr_rss(img_rss, noise_cov)

    plt.figure()
    plt.imshow(np.abs(snr_rss), cmap='jet', aspect='equal')
    plt.title('SNR map from RSS Combination')
    plt.colorbar()
    plt.savefig(f'{fname_image}_snr_rss.png')


if __name__ == '__main__':
    main()