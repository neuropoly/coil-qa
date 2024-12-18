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

from utils.utils import compute_noise_stats, reconstruct_coil_images, combine_rss, calculate_snr_rss


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Compute SNR, noise correlation, and g-factor maps.')
    parser.add_argument('fname_image', type=str, help='Path to the raw image data file')
    parser.add_argument('fname_noise', type=str, help='Path to the raw noise data file')
    # add the possibility to specify another set of image and noise data
    parser.add_argument('--other-image', type=str, help='Path to the raw image data file of another coil part')
    parser.add_argument('--other-noise', type=str, help='Path to the raw noise data file of another coil part')
    # add scaling factor, defaulted to 100
    parser.add_argument('--scaling-factor', type=float, default=100, help='Scaling factor for the SNR map')
    # Add argument to specify the maximum value of the colorbar. No default value.
    parser.add_argument('--max-colorbar', type=float, help='Maximum value for the colorbar')
    # Add argument to skip every other xtick because they overlap
    parser.add_argument('--skip-xticks', action='store_true', help='Skip every other xtick')

    args = parser.parse_args()

    fname_image = args.fname_image
    fname_noise = args.fname_noise
    fname_image2 = args.other_image
    fname_noise2 = args.other_noise

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
        # TODO: do not hardcode path
        sys.path.append('/Users/julien/code/orchestra-sdk-2.1-1')  # https://github.com/neuropoly/coil-qa/issues/2
        from GERecon import Pfile
        # Read GE 'p-file' data
        pfile = Pfile(fname_image)
        metadata = pfile.MetaData()
        meas_image = pfile.KSpace(0, 0)  # TODO: replace with (nslice, echo)
        pfile = Pfile(fname_noise)
        meas_noise = pfile.KSpace(0, 0)  # TODO: replace with (nslice, echo)
        # In case use provided another set of image and noise data, read them
        if fname_image2 is not None:
            pfile = Pfile(fname_image2)
            meas_image2 = pfile.KSpace(0, 0)
            if fname_noise2 is not None:
                pfile = Pfile(fname_noise2)
                meas_noise2 = pfile.KSpace(0, 0)
            else:
                # If no noise data is provided, throw an error
                raise ValueError('Noise data for the second image is missing')
            # Concatenate the data
            meas_image = np.concatenate((meas_image, meas_image2), axis=2)
            meas_noise = np.concatenate((meas_noise, meas_noise2), axis=2)

        # Fetch pixel size in mm
        header = pfile.Header()
        fov = header['rdb_hdr_image']['dfov']
        xdim = fov / metadata['acquiredXRes']
        ydim = fov / metadata['acquiredYRes']

    # Compute noise statistics
    noise_corr, mean_noise_corr_upper_scaled, noise_cov = compute_noise_stats(meas_noise)
    mean_noise_corr_upper_scaled *= 100  # Convert to percentage
    print(f' ==> average off-diagonal coupling: {mean_noise_corr_upper_scaled:.2f}%')

    # Display noise correlation and covariance matrices
    for matrix, title, fname_suffix in zip([noise_corr, noise_cov], 
                                           [f'Noise Correlation Matrix ({mean_noise_corr_upper_scaled:.2f}%)', 'Noise Covariance Matrix'], 
                                           ['noisecorr', 'noisecov']):
        plt.figure()
        plt.imshow(np.abs(matrix), cmap='jet', interpolation='nearest')
        # change label values so that the index starts at 1
        plt.xticks(np.arange(matrix.shape[0]), np.arange(1, matrix.shape[0] + 1))
        plt.yticks(np.arange(matrix.shape[0]), np.arange(1, matrix.shape[0] + 1))
        if args.skip_xticks:
            plt.xticks(np.arange(0, matrix.shape[0], 2), np.arange(1, matrix.shape[0] + 1, 2))
        plt.colorbar()
        plt.title(title)
        plt.savefig(f'{fname_image}_{fname_suffix}.png')

    # Reconstruct coil sensitivity images
    coil_images = reconstruct_coil_images(meas_image)

    # Combine images using RSS
    img_rss = combine_rss(coil_images)

    # Calculate the extent of the image
    Nx, Ny = img_rss.shape  # Image dimensions
    extent = [0, Nx * xdim, 0, Ny * ydim]  # [xmin, xmax, ymin, ymax]
    # display the image
    plt.figure()
    # Plot the image with scaled axes
    plt.imshow(img_rss, cmap='gray', extent=extent)
    plt.axis('off')
    plt.title('Image After RSS Combination')
    plt.colorbar()
    plt.savefig(f'{fname_image}_rss.png')

    # Calculate SNR map
    snr_rss = calculate_snr_rss(img_rss, noise_cov)

    # Multiply by scaling factor
    snr_rss *= args.scaling_factor

    plt.figure()
    # display SNR map, and colorbar with max value specified by user, if provided
    if args.max_colorbar is not None:
        plt.imshow(np.abs(snr_rss), cmap='jet', extent=extent, vmax=args.max_colorbar)
    else:
        plt.imshow(np.abs(snr_rss), cmap='jet', extent=extent)
    plt.axis('off')
    plt.title('SNR map from RSS Combination')
    plt.colorbar()
    plt.savefig(f'{fname_image}_snr_rss.png')


if __name__ == '__main__':
    main()