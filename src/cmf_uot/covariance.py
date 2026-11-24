"""Sample spatial covariance matrix at one frequency bin."""
import numpy as np
import scipy.fft as fft


def calculate_sigma(Sigs, Lseg, freq):
    """
    Calculates the spatial covariance matrix Sigma from the input signals.

    Args:
        Sigs (numpy.ndarray): The input signals (time series data).
        Lseg (int): The length of each segment for FFT.
        freq (float): The frequency of interest.

    Returns:
        numpy.ndarray: The spatial covariance matrix Sigma.
    """
    srinv = 9 # fs = 500kHz/(srinv+1), 9 <= srinv <= 255
    fs = 500000/(srinv+1)
    print(f"Sampling frequency: {fs} Hz")

    # Determine the number of segments
    Nseg = Sigs.shape[0] // Lseg
    
    # Truncate signals to an integer number of segments
    Sigs = Sigs[:Nseg*Lseg]
    
    # Find the FFT index closest to the desired frequency
    idx = int(freq / fs * Lseg)
    
    # Reshape the signals into segments
    Xseg = np.reshape(Sigs, [Nseg, Lseg, Sigs.shape[1]])
    print(f"S Sigs.shape[1]: {Sigs.shape[1]}")
    
    # Perform FFT on each segment
    XF = fft.fft(Xseg, axis=1)
    
    # Extract the data at the desired frequency
    data = XF[:, idx, :].T
    
    # Calculate the covariance matrix
    Sigma = data @ data.T.conj() / Nseg
    
    return Sigma
