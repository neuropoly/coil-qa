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
    elif fname_image.endswith('.7'):
        vendor = 'ge'
    else:
        raise ValueError('Unknown file extension')

    # Read proprietary raw 'image' data
    if vendor == 'siemens':
        meas_image = read_meas_dat(fname_image)
        meas_noise = read_meas_dat(fname_noise)
    elif vendor == 'ge':
        # Read GE 'p-file' data
        pass

    # # Display dimensions of image and noise data
    # mrir_ice_dimensions(meas_image['data'])
    # mrir_ice_dimensions(meas_noise['data'])

    # # Step 1a: Calculate channel noise correlation coefficient matrix
    # noisecof = mrir_array_stats_matrix(meas_noise['data'], 'cof')

    # # Calculate average off-diagonal coupling
    # mask = np.tril(np.ones(noisecof.shape), -1) > 0
    # noisecof_avg = np.mean(np.abs(noisecof[mask]))

    # print(f' ==> average off-diagonal coupling: {noisecof_avg * 100:.2f}%')

    # # Display noise correlation matrix
    # plt.imshow(noisecof, cmap='hot', interpolation='nearest')
    # plt.colorbar()
    # plt.title('Noise Correlation Matrix')
    # plt.show()

    # # Step 1b: Calculate channel noise covariance matrix
    # noisecov = mrir_array_stats_matrix(meas_noise['data'], 'cov', 1)

    # # Display noise covariance matrix
    # plt.figure()
    # plt.imshow(np.abs(noisecov), aspect='equal', cmap='viridis')
    # plt.title('Channel Noise Covariance Matrix')
    # plt.colorbar()
    # plt.show()

    # # Step 2: Reconstruction of coil sensitivity images
    # img = mrir_conventional_2d(meas_image['data'])
    # sens = img

    # # Step 3: Combine the image using the root-sum-of-squares method
    # img_rss = mrir_array_combine_rss(img)
    # img_rss = np.squeeze(img_rss[..., 0])

    # plt.figure()
    # plt.imshow(img_rss, cmap='gray', aspect='equal')
    # plt.title('Image After RSS Combination')
    # plt.colorbar()
    # plt.show()

    # # Step 4: Calculate corresponding SNR maps for the RSS combination method
    # snr_rss = mrir_array_SNR_rss(img, noisecov)

    # plt.figure()
    # plt.imshow(snr_rss, cmap='jet', aspect='equal', vmin=0, vmax=500)
    # plt.title('SNR of RSS Combination')
    # plt.colorbar()
    # plt.savefig(f'snr_{meas_image["file"]}.png')
    # plt.show()

if __name__ == '__main__':
    main()
