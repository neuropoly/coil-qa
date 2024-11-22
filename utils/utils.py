# This file contains utility functions for data processing and visualization.
# 
# A lot of these functions were originally written by Jon Polimeni and converted to Python by Julien Cohen-Adad.
# Terms and conditions for use, reproduction, distribution and contribution are found here:
# https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense

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
    """
    # TODO: support do_noise_bw_scaling
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

    # convert tuple to ndarray
    # result = np.array(result)

    # return result, noise if do_noise_bw_scaling else result
    return result


def mrir_conventional_2d(raw, prot=None):
    """
    Reconstructs conventional (Cartesian) acquisitions.

    Parameters:
        raw (numpy.ndarray): Raw k-space data.
        prot (dict, optional): Protocol structure containing acquisition parameters. 
                               If None, defaults will be used.

    Returns:
        img_peft (numpy.ndarray): The reconstructed image.
        prot (dict): The protocol used for reconstruction.
    """
    # Default settings
    DO_IMAGE_CROP = False  # Julien Cohen-Adad: Switched to False

    # Handle protocol input
    if prot is None:
        prot = read_meas_prot__struct()
    elif isinstance(prot, (int, float)) and prot == 0:
        prot = {}
        DO_IMAGE_CROP = False

    # Frequency encoding inverse Fourier transform
    hyb_roft = mrir_iDFT_freqencode(raw, prot.get('iNoOfFourierColumns', raw.shape[0]))

    # Phase encoding inverse Fourier transform
    img_peft = mrir_iDFT_phasencode(hyb_roft, 'lin', prot.get('iNoOfFourierLines', raw.shape[1]))

    # Optional image cropping
    if DO_IMAGE_CROP:
        img_peft = mrir_image_crop(img_peft, prot.get('flReadoutOSFactor', 1))

    return img_peft, prot


def read_meas_prot__struct(*args):
    """
    Reads protocol information and returns a structured dictionary.

    Returns:
        YAPS (dict): A dictionary containing protocol parameters with default values.
        param_list (list): A list of protocol parameter names.
    """
    # List of protocol parameters
    param_list = [
        "file", "ulVersion", "ManufacturersModelName", "InstitutionName", "tSequenceFileName", 
        "tProtocolName", "tPatientPosition", "flNominalB0", "lFrequency", 
        "flReferenceAmplitude", "flAmplitude", "lBaseResolution", "lPhaseEncodingLines", 
        "iNoOfFourierColumns", "iNoOfFourierLines", "iNoOfFourierPartitions", "uc2DInterpolation", 
        "lAccelFactPE", "lAccelFact3D", "lRefLinesPE", "lRefLines3D", "ucPATMode", 
        "ucRefScanMode", "ulCaipirinhaMode", "lReorderingShift3D", "CaipirinhaShift", 
        "ucAsymmetricEchoAllowed", "ucAsymmetricEchoMode", "flAsymmetry_DERIVED", 
        "ucPhasePartialFourier", "ucSlicePartialFourier", "alTR", "alTI", "alTE", "alTS", 
        "alDwellTime", "flBandwidthPerPixelPhaseEncode", "iEffectiveEpiEchoSpacing", 
        "lEPIFactor", "adFlipAngleDegree", "lRepetitions", "lAverages", "lContrasts", 
        "dPhaseResolution", "dSliceResolution", "lPartitions", "lNoOfPhaseCorrScans", 
        "flReadoutOSFactor", "dPhaseOversamplingForDialog", "dSliceOversamplingForDialog", 
        "tGradientCoil", "tCoilID", "sCoilElementID_tElement", "alRegridRampupTime", 
        "alRegridRampdownTime", "alRegridFlattopTime", "alRegridDelaySamplesTime", 
        "aflRegridADCDuration", "alRegridMode", "ucRegridMode", "dThickness", 
        "dPhaseFOV", "dReadoutFOV", "sSliceArray_lSize", "sSliceArray", 
        "sSliceArray_reordered", "ucMultiSliceMode", "sSliceArray_ucMode", 
        "sDiffusion_lDiffDirections", "sDiffusion_alBValue", "sWiPMemBlock_alFree", 
        "sWiPMemBlock_adFree"
    ]

    # Create a dictionary with default None values
    YAPS = {param.replace('.', '_'): None for param in param_list}

    # return YAPS, param_list
    return YAPS  # Julien Cohen-Adad: removed param_list from return


def mrir_iDFT_freqencode(raw, Npoint=None):
    """
    Performs the inverse Discrete Fourier Transform (iDFT) along the frequency encoding dimension.

    Parameters:
        raw (numpy.ndarray): The raw k-space data.
        Npoint (int, optional): Number of points for the iDFT. If None, the size of the first dimension of `raw` is used.

    Returns:
        roft (numpy.ndarray): The transformed data after iDFT along the frequency encoding dimension.
    """
    # Default Npoint to the size of the first dimension of raw if not provided
    if Npoint is None:
        Npoint = raw.shape[0]

    # Perform inverse DFT along the frequency encoding dimension
    roft = mrir_iDFT(raw, dim=0, Npoint=Npoint)

    return roft


def mrir_iDFT_phasencode(raw, coordinate_str='lin', Npoint=None):
    """
    Performs the inverse Discrete Fourier Transform (iDFT) along the phase encoding dimension.

    Parameters:
        raw (numpy.ndarray): The raw k-space data.
        coordinate_str (str, optional): Coordinate string ('lin' for linear, 'par' for parallel). Default is 'lin'.
        Npoint (int, optional): Number of points for the iDFT. If None, the size of the specified dimension of `raw` is used.

    Returns:
        peft (numpy.ndarray): The transformed data after iDFT along the phase encoding dimension.
    """
    # Default coordinate string is 'lin' (linear)
    coordinate_str = coordinate_str.lower()

    # Determine the dimension based on the coordinate string
    if coordinate_str.startswith('lin'):
        dim = 1  # Second dimension (index 1 in Python)
    elif coordinate_str.startswith('par'):
        dim = 8  # Ninth dimension (index 8 in Python)
    else:
        raise ValueError(f'Unrecognized data dimension: "{coordinate_str}"')

    # Warn if the selected dimension has size 1
    if raw.shape[dim] == 1:
        print(f'Warning: Input data has no significant size along dimension "{coordinate_str}".')

    # Default Npoint to the size of the specified dimension if not provided
    if Npoint is None:
        Npoint = raw.shape[dim]

    # Perform inverse DFT along the specified dimension
    peft = mrir_iDFT(raw, dim, Npoint=Npoint)

    return peft


def mrir_iDFT(raw, dim, Npoint=None, FLAG_siemens_style=False):
    """
    Performs the inverse Discrete Fourier Transform (iDFT) along a specified dimension.

    Parameters:
        raw (numpy.ndarray): The raw k-space data.
        dim (int): The dimension along which to perform the iDFT.
        Npoint (int, optional): Number of points for the iDFT. Defaults to the size of the specified dimension.
        FLAG_siemens_style (bool, optional): Flag for Siemens-style FFT processing. Default is False.

    Returns:
        ft (numpy.ndarray): The transformed data after iDFT.
    """
    # Default Npoint to the size of the specified dimension if not provided
    if Npoint is None:
        Npoint = raw.shape[dim]

    # Zero-padding if Npoint is greater than the size along the specified dimension
    if Npoint > raw.shape[dim]:
        pad_dims = [(0, 0)] * raw.ndim  # Create a padding specification for all dimensions
        pad_dims[dim] = (0, Npoint - raw.shape[dim])  # Pad only the specified dimension
        raw = np.pad(raw, pad_dims, mode='constant')


    # Display the k-space data
    # plt.figure()
    # plt.imshow(np.log(1 + np.abs(raw[:, :, 8])), cmap="gray")
    # plt.title("K-Space Data")
    # plt.colorbar()
    # plt.savefig("kspace_data.png")

    FLAG_ge_style = True  # Julien Cohen-Adad

    # Perform the iDFT
    if FLAG_siemens_style:
        # Siemens-style FFT
        ft = np.fft.fftshift(
            np.fft.fft(
                np.fft.ifftshift(raw, axes=dim), n=Npoint, axis=dim
            ),
            axes=dim
        )
    elif FLAG_ge_style:
        # GE-style FFT
        # TODO: make sure the axes below work for all scenario
        ft = np.fft.ifftshift(raw, axes=(0, 1))
        ft = np.fft.ifft(ft, n=Npoint, axis=dim)
        ft = np.fft.fftshift(ft, axes=(0))
    else:
        # Standard FFT
        ft = np.fft.ifftshift(raw, axes=dim)
        ft = np.fft.ifft(ft, n=Npoint, axis=dim)
        ft = np.fft.fftshift(ft, axes=dim) * Npoint

    return ft


def mrir_image_crop(img_oversampled, prot=None, os_factor_freqencode=2, os_factor_phasencode=1, os_factor_partencode=1, verbose=False):
    """
    Crop an image volume reconstructed from oversampled k-space data.

    Parameters:
        img_oversampled (numpy.ndarray): Oversampled image volume.
        prot (dict, optional): Protocol dictionary containing oversampling factors. Defaults to None.
        os_factor_freqencode (int, optional): Oversampling factor along the frequency-encoded direction. Defaults to 2.
        os_factor_phasencode (int, optional): Oversampling factor along the phase-encoded direction. Defaults to 1.
        os_factor_partencode (int, optional): Oversampling factor along the partition-encoded direction. Defaults to 1.
        verbose (bool, optional): Verbosity flag. Defaults to False.

    Returns:
        img_cropped (numpy.ndarray): Cropped image volume.
    """
    # Extract oversampling factors from the protocol if provided
    if prot is not None:
        os_factor_freqencode = prot.get("flReadoutOSFactor", os_factor_freqencode)
        os_factor_phasencode = prot.get("dPhaseOversamplingForDialog", os_factor_phasencode - 1) + 1
        os_factor_partencode = prot.get("dSliceOversamplingForDialog", os_factor_partencode - 1) + 1

    dims = img_oversampled.shape

    # Ensure oversampling factors are valid
    for dim, os_factor, dim_name in zip(
        [0, 1, 2],
        [os_factor_freqencode, os_factor_phasencode, os_factor_partencode],
        ["frequency-encoded", "phase-encoded", "partition-encoded"]
    ):
        if os_factor > 1 and (dims[dim] % os_factor != 0):
            raise ValueError(f"Samples are not integer multiples of the oversampling factor along the {dim_name} direction.")

    img_cropped = img_oversampled.copy()

    # Crop along the frequency-encoded direction
    if os_factor_freqencode > 1:
        newdim = dims[0] // os_factor_freqencode
        start_index = (dims[0] - newdim) // 2
        end_index = start_index + newdim
        if verbose:
            print(f"Cropping FREQENCODE dimension: {dims[0]} -> {newdim}")
        img_cropped = img_cropped[start_index:end_index, ...]

    # Crop along the phase-encoded direction
    if os_factor_phasencode > 1:
        newdim = dims[1] // os_factor_phasencode
        start_index = (dims[1] - newdim) // 2
        end_index = start_index + newdim
        if verbose:
            print(f"Cropping PHASENCODE dimension: {dims[1]} -> {newdim}")
        img_cropped = img_cropped[:, start_index:end_index, ...]

    # Crop along the partition-encoded direction
    if os_factor_partencode > 1:
        newdim = dims[2] // os_factor_partencode
        start_index = (dims[2] - newdim) // 2
        end_index = start_index + newdim
        if verbose:
            print(f"Cropping PARTENCODE dimension: {dims[2]} -> {newdim}")
        img_cropped = img_cropped[:, :, start_index:end_index, ...]

    return img_cropped


def mrir_array_combine_rss(img_multichan):
    """
    Combines multi-channel image data using the Root-Sum-of-Squares (RSS) method.

    Parameters:
        img_multichan (numpy.ndarray): Multi-channel image data with the channel dimension as the third axis.

    Returns:
        img_combine_rss (numpy.ndarray): Image combined using the RSS method.
    """
    # Compute the root-sum-of-squares across the channel dimension (axis=2)
    img_combine_rss = np.sqrt(np.sum(np.abs(img_multichan) ** 2, axis=2))

    return img_combine_rss


import numpy as np
import matplotlib.pyplot as plt
